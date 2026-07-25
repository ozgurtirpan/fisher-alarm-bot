"""
FISHER TRANSFORM + TELEGRAM ALARM BOTU (GitHub Actions sürümü)
================================================================
Bu sürüm, bilgisayarında sürekli açık kalan bir program yerine,
GitHub Actions tarafından periyodik olarak (örn. her 15 dakikada)
tetiklenip BİR KERE çalışıp kapanacak şekilde tasarlandı.

Fark:
- Sonsuz döngü (while True) YOK, tek seferlik kontrol yapıp kapanıyor
- Telegram bilgileri koda yazılmıyor, GitHub Secrets üzerinden
  ortam değişkeni (environment variable) olarak geliyor
- Hangi pariteye ne zaman alarm gönderildiği, tekrar aynı sinyali
  göndermemek için state.json dosyasında saklanıyor ve her çalışmada
  GitHub'a geri kaydediliyor (workflow dosyası bunu otomatik yapıyor)
"""

import os
import json
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ============== AYARLAR ==============

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

PAIRS = {
    "NATGAS": "NG=F",
    "AVAXUSD": "AVAX-USD",
}

FETCH_INTERVAL = "1h"      # Yahoo Finance'ten çekilecek ham veri periyodu (4h Yahoo'da doğrudan yok)
LOOKBACK_PERIOD = "60d"    # 4 saatlik mumlar için yeterli geçmiş veri (60 gün ~ 360 adet 4h mum)
RESAMPLE_TO = "4h"         # Ham 1 saatlik veriyi bu periyoda grupluyoruz
FISHER_LENGTH = 9
STATE_FILE = "state.json"

# =======================================


def resample_to_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """1 saatlik ham veriyi istenen periyoda (örn. 4 saat) gruplar."""
    resampled = df.resample(target).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    return resampled


def fisher_transform(df: pd.DataFrame, length: int = 9) -> pd.DataFrame:
    median_price = (df["High"] + df["Low"]) / 2
    max_high = median_price.rolling(window=length).max()
    min_low = median_price.rolling(window=length).min()
    value_range = (max_high - min_low).replace(0, 1e-10)

    raw_value = 2 * ((median_price - min_low) / value_range - 0.5)
    raw_value = raw_value.clip(-0.999, 0.999)

    smoothed = pd.Series(index=df.index, dtype=float)
    fisher = pd.Series(index=df.index, dtype=float)

    prev_smoothed = 0.0
    prev_fisher = 0.0

    for i in range(len(df)):
        rv = raw_value.iloc[i]
        if pd.isna(rv):
            smoothed.iloc[i] = 0.0
            fisher.iloc[i] = 0.0
            continue
        curr_smoothed = max(min(0.33 * rv + 0.67 * prev_smoothed, 0.999), -0.999)
        curr_fisher = 0.5 * np.log((1 + curr_smoothed) / (1 - curr_smoothed)) + 0.5 * prev_fisher
        smoothed.iloc[i] = curr_smoothed
        fisher.iloc[i] = curr_fisher
        prev_smoothed = curr_smoothed
        prev_fisher = curr_fisher

    df = df.copy()
    df["fisher"] = fisher
    df["signal"] = fisher.shift(1)
    return df


def detect_crossover(df: pd.DataFrame):
    if len(df) < 3:
        return None
    prev_fisher, prev_signal = df["fisher"].iloc[-2], df["signal"].iloc[-2]
    curr_fisher, curr_signal = df["fisher"].iloc[-1], df["signal"].iloc[-1]
    if pd.isna(prev_fisher) or pd.isna(curr_fisher):
        return None
    if prev_fisher <= prev_signal and curr_fisher > curr_signal:
        return "YUKARI KESİŞİM (potansiyel ALIŞ)"
    if prev_fisher >= prev_signal and curr_fisher < curr_signal:
        return "AŞAĞI KESİŞİM (potansiyel SATIŞ)"
    return None


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[HATA] TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID tanımlı değil (GitHub Secrets kontrol et).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"[UYARI] Telegram mesajı gönderilemedi: {response.text}")
    except Exception as e:
        print(f"[HATA] Telegram gönderim hatası: {e}")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def check_pair(pair_name: str, ticker: str, state: dict) -> None:
    try:
        raw_data = yf.download(ticker, period=LOOKBACK_PERIOD, interval=FETCH_INTERVAL, progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        if raw_data.empty:
            print(f"[{pair_name}] Yetersiz veri, atlanıyor.")
            return

        data = resample_to_target(raw_data, RESAMPLE_TO)

        if data.empty or len(data) < FISHER_LENGTH + 2:
            print(f"[{pair_name}] Resample sonrası yetersiz veri, atlanıyor.")
            return

        data = fisher_transform(data, FISHER_LENGTH)
        signal = detect_crossover(data)
        last_bar_time = str(data.index[-1])

        if signal and state.get(pair_name) != last_bar_time:
            message = (
                f"📊 {pair_name} - Fisher Transform Sinyali\n"
                f"{signal}\n"
                f"Zaman: {last_bar_time}\n"
                f"Fiyat: {float(data['Close'].iloc[-1]):.5f}\n\n"
                f"⚠️ Bu otomatik bir alım/satım emri değildir, sadece bir uyarıdır."
            )
            send_telegram_message(message)
            state[pair_name] = last_bar_time
            print(f"[{pair_name}] {signal} -> Telegram'a gönderildi.")
        else:
            print(f"[{pair_name}] Kesişim yok. Son fisher: {data['fisher'].iloc[-1]:.3f}")

    except Exception as e:
        print(f"[{pair_name}] HATA: {e}")


def main() -> None:
    print(f"Kontrol zamanı: {datetime.datetime.now()}")
    state = load_state()
    for pair_name, ticker in PAIRS.items():
        check_pair(pair_name, ticker, state)
    save_state(state)
    print("Kontrol tamamlandı.")


if __name__ == "__main__":
    main()
