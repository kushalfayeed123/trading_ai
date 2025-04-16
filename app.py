import asyncio
from datetime import datetime, timezone
import logging
import logging.handlers
import json
import math
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from ta.trend import ADXIndicator, MACD
from ta.momentum import StochasticOscillator, RSIIndicator, ROCIndicator
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import RobustScaler
from deriv_api import DerivAPI  # Ensure you have the latest Deriv API package
from flask import Flask, request, jsonify
from flask_cors import CORS
import nest_asyncio
import threading
import requests
import os
import joblib
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()  # load variables from .env file

api_token = os.environ.get("API_TOKEN")
app_id = os.environ.get("APP_ID")

trading_enabled = False
bot_instance = None
notification_msg = ""

# Global feature columns (feel free to add more as you experiment)
FEATURE_COLS = [
    'rsi', 'rsi_diff', 'macd', 'macd_signal', 'macd_hist',
    'bollinger_hband', 'bollinger_lband', 'stoch_k',
    'roc', 'ma_20', 'ma_50', 'ma_diff', 'atr', 'adx', 'regime',
    'volatility_ratio', 'dist_to_support', 'dist_to_resistance'
]


def safe_indicator_output(ind_val, index):
    try:
        arr = np.array(ind_val).flatten()
        if len(arr) == 0:
            return pd.Series(np.zeros(len(index)), index=index)
        return pd.Series(arr, index=index[-len(arr):])
    except Exception as e:
        logger.error(f"safe_indicator_output error: {e}")
        return pd.Series(np.zeros(len(index)), index=index)


def fetch_market_sentiment():
    """
    Fetch a market sentiment score from an external API.
    Returns a normalized sentiment between -1 (extreme fear) and 1 (extreme greed).
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                sentiment_value = float(data["data"][0]["value"])  # 0 to 100
                normalized = (sentiment_value - 50) / 50.0
                return normalized
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching market sentiment: {e}")
        return 0.0


def setup_logger(name: str, log_file: str, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='D', interval=1, backupCount=7)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


logger = setup_logger("EnhancedTradingBot", "enhanced_trading_bot.log")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/start", methods=["POST"])
def start_trading():
    global trading_enabled, notification_msg
    trading_enabled = True
    notification_msg = ""
    logger.info("Trading manually started via web interface.")
    return jsonify({"status": "Trading started"}), 200


@app.route("/stop", methods=["POST"])
def stop_trading():
    global trading_enabled
    trading_enabled = False
    logger.info("Trading manually stopped via web interface.")
    return jsonify({"status": "Trading stopped"}), 200


@app.route("/set_duration", methods=["POST"])
def set_duration():
    global bot_instance
    data = request.get_json()
    new_duration = data.get("duration")
    if new_duration is not None:
        bot_instance.contract_duration = int(new_duration)
        logger.info(f"Contract duration updated to: {new_duration} minutes")
        return jsonify({"status": "Contract duration updated", "duration": new_duration}), 200
    else:
        return jsonify({"error": "No duration provided"}), 400


@app.route("/set_symbol", methods=["POST"])
def set_symbol():
    global bot_instance
    data = request.get_json()
    new_symbol = data.get("symbol")
    if new_symbol is not None:
        bot_instance.training_symbol = new_symbol
        logger.info(f"Trading symbol updated to: {new_symbol}")
        return jsonify({"status": "Trading symbol updated", "symbol": new_symbol}), 200
    else:
        return jsonify({"error": "No symbol provided"}), 400


@app.route("/market_data", methods=["GET"])
def market_data():
    symbol = request.args.get("symbol", "EURUSD=X")
    interval = request.args.get("interval", "1m")
    try:
        if symbol.lower().startswith("frx"):
            symbol = symbol[3:] + "=X"
            df = yf.download(symbol, period="1d",
                             interval=interval, auto_adjust=True)
        elif symbol.upper() in ["BOOM1000", "CRASH1000"]:
            async def fetch_data():
                req = {
                    "ticks_history": symbol.upper(),
                    "count": 1440,
                    "end": "latest",
                    "granularity": 60,
                }
                try:
                    response = await bot_instance.api.send(req)
                    if ("history" in response and
                        "prices" in response["history"] and
                            "times" in response["history"]):
                        df = pd.DataFrame({
                            "Datetime": pd.to_datetime(response["history"]["times"], unit="s", utc=True),
                            "Price": response["history"]["prices"]
                        })
                        df.set_index("Datetime", inplace=True)
                        return df
                    else:
                        raise Exception(
                            "Expected tick history data not in response")
                except Exception as e:
                    logger.error(f"Error fetching boom/crash data: {e}")
                    raise
            future = asyncio.run_coroutine_threadsafe(
                fetch_data(), bot_instance.loop)
            df = future.result(timeout=10)
        else:
            df = yf.download(symbol, period="1d",
                             interval=interval, auto_adjust=True)
        df.reset_index(inplace=True)
        df.columns = ['_'.join(map(str, col)) if isinstance(
            col, tuple) else str(col) for col in df.columns]
        data = df.to_dict(orient="records")
        return jsonify({"data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    global trading_enabled
    status_data = {
        "trading_enabled": trading_enabled,
        "capital": bot_instance.capital if bot_instance and bot_instance.capital is not None else "N/A",
        "fixed_stake": bot_instance.fixed_stake if bot_instance and bot_instance.fixed_stake is not None else "N/A",
        "cycle_count": bot_instance.cycle_count if bot_instance else "N/A",
        "trade_history": bot_instance.trade_history if bot_instance and hasattr(bot_instance, "trade_history") else [],
        "cumulative_pnl": bot_instance.cumulative_pnl if hasattr(bot_instance, "cumulative_pnl") else 0.0,
        "trading_symbol": bot_instance.training_symbol if bot_instance else "N/A",
        "contract_duration": bot_instance.contract_duration if bot_instance else "N/A"
    }
    return jsonify(status_data)


@app.route("/notification", methods=["GET"])
def notification():
    return jsonify({"notification": notification_msg})


def detect_support_resistance(df, short_window=50, long_window=200):
    if len(df) < long_window:
        return None, None
    df["rolling_min"] = df["price"].rolling(window=short_window).min()
    df["rolling_max"] = df["price"].rolling(window=short_window).max()
    short_support = df["rolling_min"].iloc[-1]
    short_resistance = df["rolling_max"].iloc[-1]
    long_support = df["price"].rolling(window=long_window).min().iloc[-1]
    long_resistance = df["price"].rolling(window=long_window).max().iloc[-1]
    return (short_support, short_resistance), (long_support, long_resistance)


def sophisticated_support_resistance(df, window=10, tolerance=0.002):
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    if "price" in df.columns:
        price_col = "price"
    elif "close" in df.columns:
        price_col = "close"
    else:
        raise KeyError(
            "Expected 'price' or 'close' column not found in DataFrame")
    prices = df[price_col].values
    swing_lows = []
    swing_highs = []
    for i in range(window, len(prices) - window):
        window_slice = prices[i - window: i + window + 1]
        current = prices[i]
        if current == window_slice.min():
            swing_lows.append(current)
        if current == window_slice.max():
            swing_highs.append(current)
    if not swing_lows or not swing_highs:
        return None, None
    swing_lows.sort()
    clusters_low = []
    current_cluster = [swing_lows[0]]
    for price in swing_lows[1:]:
        if abs(price - current_cluster[-1]) / current_cluster[-1] < tolerance:
            current_cluster.append(price)
        else:
            clusters_low.append(current_cluster)
            current_cluster = [price]
    clusters_low.append(current_cluster)
    support = np.mean(max(clusters_low, key=len))
    swing_highs.sort(reverse=True)
    clusters_high = []
    current_cluster = [swing_highs[0]]
    for price in swing_highs[1:]:
        if abs(price - current_cluster[-1]) / current_cluster[-1] < tolerance:
            current_cluster.append(price)
        else:
            clusters_high.append(current_cluster)
            current_cluster = [price]
    clusters_high.append(current_cluster)
    resistance = np.mean(max(clusters_high, key=len))
    return support, resistance


def should_trade(price, trade_signal, short_sr, long_sr):
    buffer_ratio = 0.005  # 0.5% buffer
    if trade_signal == "CALL":
        if price <= short_sr * (1 + buffer_ratio):
            return True
    elif trade_signal == "PUT":
        if price >= short_sr * (1 - buffer_ratio):
            return True
    return False


class DerivTradingBot:
    def __init__(self, app_id: str, api_token: str, training_symbol: str = "frxEURUSD", contract_duration: int = 30):

        self.FEATURE_COLS = [
            'rsi', 'rsi_diff', 'macd', 'macd_signal', 'macd_hist',
            'bollinger_hband', 'bollinger_lband', 'stoch_k',
            'roc', 'ma_20', 'ma_50', 'ma_diff', 'atr', 'adx',
            'regime', 'volatility_ratio', 'dist_to_support', 'dist_to_resistance',
            'sentiment'
        ]
        self.app_id = app_id
        self.api_token = api_token
        self.training_symbol = training_symbol
        self.contract_duration = contract_duration
        self.api = None
        self.logger = logger

        # Capital & Risk Management
        self.capital = None
        self.fixed_stake = None
        self.min_stake = 0.50
        self.max_stake = 1

        self.current_position = None  # 1 for CALL, 0 for PUT
        self.hysteresis_margin = 0.1

        self.training_iterations = 0
        self.MIN_TRAINING_CYCLES = 50

        self.scaler = None

        self.experience_buffer = []
        self.last_trade_state = None
        self.last_trade_action = None

        self.train_window = 1000
        self.retrain_freq = 60  # seconds

        self.cycle_count = 0
        self.cumulative_pnl = 0.0
        self.consecutive_loss_count = 0
        self.trade_history = []
        self.last_trade_record = None
        self.vol_threshold = 0.005  # sensible default
        self.symbol = None
        self.vol_threshold = 0.005
        self.confidence_threshold = 0.5
        self.current_position = None
        self.feature_names = self.FEATURE_COLS.copy()
        self.model_initialized = False
        self.training_iterations = 0
        self.cycle_count = 0
        self.trade_history = []

        # Advanced model using kernel approximation for non-linearity
        self.use_advanced_model = True

        self.ml_model = SGDClassifier(
            loss='log_loss',            # Enables predict_proba
            alpha=0.001,
            learning_rate='optimal',
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )

        self.model_initialized = False
        if self.use_advanced_model:
            self.rbf_sampler = RBFSampler(gamma=0.1, random_state=42)
            self.rbf_fitted = False
        self.load_persistence()

    def save_persistence(self):
        try:
            joblib.dump(self.ml_model, 'model.pkl')
            if self.scaler is not None:
                joblib.dump(self.scaler, 'scaler.pkl')
            if self.use_advanced_model and hasattr(self, 'rbf_sampler'):
                joblib.dump(self.rbf_sampler, 'rbf_sampler.pkl')
            joblib.dump(self.training_iterations, 'training_iterations.pkl')
            self.logger.info(
                "Model, preprocessing objects, and training iteration count saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving persistence files: {e}")

    def load_persistence(self):
        try:
            if os.path.exists('model.pkl'):
                self.ml_model = joblib.load('model.pkl')
                self.model_initialized = True
                self.logger.info("Loaded persisted model.")
                if not hasattr(self.ml_model, 'predict_proba'):
                    self.logger.error(
                        "Loaded model does not support predict_proba. Use SGDClassifier with loss='log_loss'.")

            if os.path.exists('scaler.pkl'):
                self.scaler = joblib.load('scaler.pkl')
                self.logger.info("Loaded persisted scaler.")
            if self.use_advanced_model and os.path.exists('rbf_sampler.pkl'):
                self.rbf_sampler = joblib.load('rbf_sampler.pkl')
                self.rbf_fitted = True
                self.logger.info("Loaded persisted RBFSampler.")
            if os.path.exists('training_iterations.pkl'):
                self.training_iterations = joblib.load(
                    'training_iterations.pkl')
                self.logger.info(
                    f"Loaded training iteration count: {self.training_iterations}")
            if os.path.exists('feature_names.pkl'):
                self.feature_names = joblib.load('feature_names.pkl')
                self.logger.info("Loaded feature names.")
            else:
                self.feature_names = self.FEATURE_COLS.copy()

        except Exception as e:
            self.logger.error(f"Error loading persistence files: {e}")

    def transform_features(self, X):
        try:
            if self.use_advanced_model:
                # Automatically refit RBF if feature count changes
                if not self.rbf_fitted or (
                    hasattr(self.rbf_sampler, "n_features_in_")
                    and X.shape[1] != self.rbf_sampler.n_features_in_
                ):
                    self.logger.warning(
                        f"Refitting RBFSampler: incoming feature count = {X.shape[1]}, previous = {getattr(self.rbf_sampler, 'n_features_in_', 'N/A')}"
                    )
                    self.rbf_sampler.fit(X)
                    self.rbf_fitted = True
                    # Persist the updated RBF if needed
                    joblib.dump(self.rbf_sampler, 'rbf_sampler.pkl')
                return self.rbf_sampler.transform(X)
            else:
                return X
        except Exception as e:
            self.logger.error(f"Error in transform_features: {e}")
            raise

    def predict_trade_signal(self, df):
        try:
            features = df.iloc[-1:].drop(columns=["Price"], errors="ignore")
            if self.scaler is None:
                return "NEUTRAL"
            features_scaled = self.scaler.transform(features)
            features_transformed = self.transform_features(features_scaled)
            prediction = self.ml_model.predict(features_transformed)[0]
            return "CALL" if prediction == 1 else "PUT"
        except Exception as e:
            self.logger.error(f"Error in predict_trade_signal: {e}")
            return "NEUTRAL"

    def add_regime_features(self, df):
        try:
            df['long_ma'] = df['close'].rolling(
                window=200, min_periods=50).mean()
            df['regime'] = np.where(df['close'] > df['long_ma'], 1, 0)
            self.logger.info("Added market regime features.")
            return df
        except Exception as e:
            self.logger.error(f"Error adding regime features: {e}")
            return df

    def preprocess_data(self, df):
        try:
            df.columns = [('_'.join(map(str, col)) if isinstance(
                col, tuple) else str(col)).lower().rstrip('_') for col in df.columns]

            if not isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
                if 'datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    df.set_index('datetime', inplace=True)

            # Rename columns for synthetic or forex assets
            if self.training_symbol.upper() in ["BOOM1000", "CRASH1000"]:
                if "price" in df.columns:
                    df["close"] = df["price"]
                    df["open"] = df["price"]
                    df["high"] = df["price"]
                    df["low"] = df["price"]
                else:
                    raise KeyError(
                        "Missing price column for synthetic indices")
            else:
                def find_and_rename_price_columns(df):
                    def find_column(target):
                        for col in df.columns:
                            if target in col:
                                return col
                        return None
                    close_col = find_column("close")
                    high_col = find_column("high")
                    low_col = find_column("low")
                    if close_col is None or high_col is None or low_col is None:
                        raise KeyError("Missing price columns")
                    df.rename(
                        columns={close_col: "close", high_col: "high", low_col: "low"}, inplace=True)
                    return df
                df = find_and_rename_price_columns(df)

            for col in ['close', 'high', 'low']:
                df[col] = pd.Series(df[col].values.ravel(), index=df.index)

            close = df['close']
            high = df['high']
            low = df['low']

            # === Feature Engineering ===
            df['rsi'] = RSIIndicator(close, window=14, fillna=True).rsi()
            df['rsi_diff'] = df['rsi'].diff().fillna(0)
            macd_obj = MACD(close, fillna=True)
            df['macd'] = macd_obj.macd()
            df['macd_signal'] = macd_obj.macd_signal()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            bb = ta.volatility.BollingerBands(close, fillna=True)
            df['bollinger_hband'] = bb.bollinger_hband()
            df['bollinger_lband'] = bb.bollinger_lband()
            df['stoch_k'] = StochasticOscillator(
                high, low, close, fillna=True).stoch()
            df['roc'] = ROCIndicator(close, window=12, fillna=True).roc()
            df['ma_20'] = close.rolling(window=20).mean()
            df['ma_50'] = close.rolling(window=50).mean()
            df['ma_diff'] = df['ma_20'] - df['ma_50']
            df['atr'] = ta.volatility.AverageTrueRange(
                high, low, close, fillna=True).average_true_range()
            df['adx'] = ADXIndicator(high, low, close, fillna=True).adx()
            df['regime'] = np.where(df['close'] > df['ma_50'], 1, 0)
            df['volatility_ratio'] = df['atr'] / df['close']

            # Sentiment (if available)
            df['sentiment'] = fetch_market_sentiment()

            # Support/Resistance
            try:
                sr_support, sr_resistance = sophisticated_support_resistance(
                    df, window=10, tolerance=0.002)
                df['dist_to_support'] = df['close'] - sr_support
                df['dist_to_resistance'] = sr_resistance - df['close']
            except Exception as e:
                self.logger.warning(f"S/R calc failed: {e}")
                df['dist_to_support'] = 0.0
                df['dist_to_resistance'] = 0.0

            # Target for training
            df['future_return'] = df['close'].shift(-1) - df['close']
            threshold_pct = 0.0001
            df['direction'] = np.where(df['future_return'] > df['close'] * threshold_pct, 1,
                                       np.where(df['future_return'] < -df['close'] * threshold_pct, 0, np.nan))

            # === Handle NaNs and Column Alignment ===
            for col in self.FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].ffill().bfill().fillna(0)

            # Drop rows where target is missing (only for training)
            if 'direction' in df.columns:
                df = df[df['direction'].notna() | df.index.isin([df.index[-1]])]

            # Final check
            if df.empty or df.shape[0] < 1:
                self.logger.error("Preprocessed DataFrame is empty.")
                return None

            self.logger.info(" Preprocessing complete with full feature set.")
            return df

        except Exception as e:
            self.logger.error(f"Error in safe preprocessing: {e}")
            return None

    def get_additional_features(self, df):
        try:
            sentiment = fetch_market_sentiment()
            df['sentiment'] = sentiment
            self.logger.info(f"Market sentiment added: {sentiment:.2f}")
            return df
        except Exception as e:
            self.logger.error(f"Error adding additional features: {e}")
            return df
        
    async def training_loop(self):
        self.loop = asyncio.get_running_loop()
        while True:
            try:
                if self.training_symbol is None:
                    self.logger.warning("No training symbol set; skipping training.")
                    await asyncio.sleep(self.retrain_freq)
                    continue

                safe_total = self.MIN_TRAINING_CYCLES if self.training_iterations < self.MIN_TRAINING_CYCLES else "inf"
                self.logger.info(f"Training iteration {self.training_iterations + 1}/{safe_total}")

                df_train = await self.fetch_historical_data(count=500, granularity=60)
                if df_train is None or df_train.empty:
                    self.logger.warning("Training data fetch returned empty.")
                    await asyncio.sleep(self.retrain_freq)
                    continue

                df_train = self.preprocess_data(df_train)
                if df_train is None or df_train.empty:
                    self.logger.warning("Preprocessed training data is empty.")
                    await asyncio.sleep(self.retrain_freq)
                    continue

                df_train = df_train[df_train['direction'].notna()]
                if df_train.empty:
                    self.logger.warning("All target labels are NaN; skipping model update.")
                    await asyncio.sleep(self.retrain_freq)
                    continue

                self.calibrate_asset_parameters(df_train)

                X_train = df_train[self.FEATURE_COLS]
                y_train = df_train['direction']

                self.feature_names = X_train.columns.tolist()
                joblib.dump(self.feature_names, 'feature_names.pkl')

                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                self.scaler = scaler
                X_train_transformed = self.transform_features(X_train_scaled)

                if not self.model_initialized:
                    self.logger.info("Initializing model with partial_fit...")
                    self.ml_model.partial_fit(X_train_transformed, y_train, classes=np.array([0, 1]))
                    self.model_initialized = True
                else:
                    self.logger.info("Updating model with partial_fit...")
                    self.ml_model.partial_fit(X_train_transformed, y_train)

                self.training_iterations += 1
                self.logger.info(f"Model updated. Training iterations: {self.training_iterations}")
                self.save_persistence()
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")

            await asyncio.sleep(self.retrain_freq)



    # async def training_loop(self):
    #     if self.training_symbol is None:
    #         self.logger.warning("No training symbol set; skipping training.")
    #         return

    #     self.logger.info(f"Training on {self.training_symbol}...")
    #     df_train = await self.fetch_historical_data(count=500, granularity=60)

    #     if df_train is None or df_train.empty:
    #         self.logger.warning("Training data fetch returned empty.")
    #         return

    #     df_train = self.preprocess_data(df_train)

    #     if df_train is None or df_train.empty:
    #         self.logger.warning("Preprocessed training data is empty.")
    #         return

    #     # Filter out rows where the target is missing
    #     df_train = df_train[df_train['direction'].notna()]
    #     if df_train.empty:
    #         self.logger.warning(
    #             "All target labels are NaN; skipping model update.")
    #         return

    #     self.calibrate_asset_parameters(df_train)

    #     X_train = df_train[self.FEATURE_COLS]
    #     y_train = df_train['direction']

    #     # Save features used during training
    #     self.feature_names = X_train.columns.tolist()
    #     joblib.dump(self.feature_names, 'feature_names.pkl')

    #     # Fit or update the model
    #     try:
    #         scaler = RobustScaler()
    #         X_train_scaled = scaler.fit_transform(X_train)
    #         self.scaler = scaler
    #         X_train_transformed = self.transform_features(X_train_scaled)

    #         if not self.model_initialized:
    #             self.logger.info("Initializing model with partial_fit...")
    #             self.ml_model.partial_fit(
    #                 X_train_transformed, y_train, classes=np.array([0, 1]))
    #             self.model_initialized = True
    #         else:
    #             self.logger.info(
    #                 "Updating online model with new data using partial_fit on advanced features.")
    #             self.ml_model.partial_fit(X_train_transformed, y_train)

    #         self.training_iterations += 1
    #         self.logger.info(
    #             f"Model updated. Training iterations: {self.training_iterations}")
    #     except Exception as e:
    #         self.logger.error(f"Error during model update: {e}")

    def calibrate_asset_parameters(self, df):
        try:
            avg_atr = df['atr'].mean()
            avg_price = df['close'].mean()
            computed_vol_ratio = avg_atr / avg_price if avg_price != 0 else 0.01
            if self.training_symbol.lower().startswith("frx"):
                self.vol_threshold = max(0.0001, min(
                    0.015, computed_vol_ratio * 1.0))
                self.confidence_threshold = 0.50
            elif self.training_symbol.upper() in ["BOOM1000", "CRASH1000"]:
                self.vol_threshold = max(0.015, min(
                    0.03, computed_vol_ratio * 1.5))
                self.confidence_threshold = 0.45
            else:
                self.vol_threshold = computed_vol_ratio * 1.0
                self.confidence_threshold = 0.50
            self.logger.info(
                f"Calibrated parameters for {self.training_symbol}: vol_threshold={self.vol_threshold:.3f}, confidence_threshold={self.confidence_threshold:.2f}")
        except Exception as e:
            self.logger.error(f"Error calibrating asset parameters: {e}")

    async def evaluate_and_trade(self):
        self.cycle_count += 1
        cycle_record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),

            "cycle": self.cycle_count,
            "decision": None,
            "confidence": None,
            "note": ""
        }

        if self.training_iterations < self.MIN_TRAINING_CYCLES:
            cycle_record["decision"] = "NEUTRAL"
            cycle_record["note"] = f"Model warming up (iterations: {self.training_iterations})."
            self.trade_history.append(cycle_record)
            self.logger.info(cycle_record["note"])
            return
        if not self.symbol:
            self.symbol = self.training_symbol or "unknown"

        self.logger.info(
            f"Starting trade cycle #{self.cycle_count} on {self.symbol}...")

        df_latest = await self.fetch_historical_data(count=100, granularity=60)

        if df_latest is None or df_latest.empty:
            cycle_record["decision"] = "SKIP"
            cycle_record["note"] = "No live data available."
            self.trade_history.append(cycle_record)
            return

        df_latest = self.preprocess_data(df_latest)
        if df_latest is None or df_latest.empty:
            cycle_record["decision"] = "SKIP"
            cycle_record["note"] = "Error preprocessing data."
            self.trade_history.append(cycle_record)
            return

        # Safe feature extraction
        try:
            latest_features = df_latest.iloc[-1:][self.feature_names]
        except Exception as e:
            self.logger.error(f"Error extracting latest features: {e}")
            cycle_record["decision"] = "SKIP"
            cycle_record["note"] = "Latest features unavailable."
            self.trade_history.append(cycle_record)
            return

        # Transform
        try:
            latest_features_scaled = self.scaler.transform(latest_features)
            latest_features_transformed = self.transform_features(
                latest_features_scaled)
        except Exception as e:
            cycle_record["decision"] = "SKIP"
            cycle_record["note"] = f"Error transforming features: {e}"
            self.trade_history.append(cycle_record)
            return

        # Predict
        try:
            probabilities = self.ml_model.predict_proba(
                latest_features_transformed)[0]
            prediction = np.argmax(probabilities)
            confidence = max(probabilities)
        except Exception as e:
            cycle_record["decision"] = "SKIP"
            cycle_record["note"] = f"Prediction error: {e}"
            self.trade_history.append(cycle_record)
            return

        trade_signal = "CALL" if prediction == 1 else "PUT"
        cycle_record["decision"] = trade_signal
        cycle_record["confidence"] = confidence

        if confidence < 0.4:
            cycle_record["note"] = f"Low confidence: {confidence:.2f}"
            self.trade_history.append(cycle_record)
            return

        # Stake adjustment based on volatility
        try:
            price = df_latest['close'].iloc[-1]
            atr = df_latest['atr'].iloc[-1]
            volatility_factor = atr / price
            adjusted_stake = self.fixed_stake * \
                (0.5 / volatility_factor) if volatility_factor > 0 else self.fixed_stake
            stake = max(min(adjusted_stake, self.max_stake), self.min_stake)
        except Exception as e:
            self.logger.error(f"Stake adjustment failed: {e}")
            stake = self.fixed_stake

        # Execute trade
        await self.place_trade(prediction, confidence, stake)
        self.current_position = prediction
        cycle_record["note"] = "Trade executed."
        self.trade_history.append(cycle_record)

        self.logger.info("Trade cycle complete.")

    async def get_account_currency(self):
        try:
            response = await self.api.send({"balance": 1})
            if "balance" in response:
                self.logger.info(response)
                return response["balance"].get("currency", "USD")
        except Exception as e:
            self.logger.error(f"Error fetching account currency: {e}")
        return "USD"

    async def place_trade(self, prediction, confidence, stake):
        decision = "CALL" if prediction == 1 else "PUT"
        currency = await self.get_account_currency()

        self.logger.info(
            f"Placing {decision} trade with {stake:.2f} USDT stake (Confidence: {confidence:.2f}).")

        proposal_req = {
            "amount": stake,
            "basis": "stake",
            "contract_type": decision,
            "currency": currency,
            "duration": self.contract_duration,
            "duration_unit": "m",
            "symbol": self.training_symbol
        }

        proposal_response = await self.api.proposal(proposal_req)
        self.logger.info(f"Proposal response: {proposal_response}")

        if not proposal_response or "proposal" not in proposal_response:
            self.logger.error("Trade proposal failed.")
            return

        proposal_id = proposal_response["proposal"].get("id")
        if not proposal_id:
            self.logger.error("Proposal ID missing; aborting trade.")
            return

        buy_msg = {"buy": proposal_id, "price": stake}
        buy_response = await self.api.buy(buy_msg)
        if buy_response.get("error"):
            self.logger.error(
                f"Trade execution failed: {buy_response['error']}")
            return

        contract_id = buy_response.get("contract_id")

        # Save for training later
        # already set in evaluate_and_trade
        self.last_trade_state = self.last_trade_state
        self.last_trade_action = prediction

        # Set price and ATR for stop loss monitoring
        df_latest = await self.fetch_historical_data(count=100, granularity=60)
        df_latest = self.preprocess_data(df_latest)
        if df_latest is not None and not df_latest.empty:
            self.last_entry_price = df_latest["close"].iloc[-1]
            self.last_entry_atr = float(df_latest["atr"].iloc[-1])

        # Monitor contract
        # asyncio.create_task(self.monitor_trade_for_stop_loss(contract_id, stake))
        asyncio.create_task(self.check_contract_result(contract_id))

    # async def monitor_trade_for_stop_loss(self, contract_id, stake):
    #     """
    #     Monitor an open contract for early exit conditions.
    #     If the profit falls below a stop-loss threshold or a trailing stop is triggered,
    #     attempt an early sell.
    #     """
    #     # Set thresholds relative to stake (adjust these as needed)
    #     # Calculate stop-loss threshold based on ATR.
    #     # For CALL: stop-loss is triggered if profit falls below a loss equivalent to 1.5 x entry ATR.
    #     # For PUT: stop-loss is triggered if profit exceeds a loss of 1.5 x entry ATR.
    #     if self.last_trade_action == 1:  # CALL
    #         stop_loss_threshold = -1.5 * self.last_entry_atr
    #     else:  # PUT
    #         stop_loss_threshold = 1.5 * self.last_entry_atr

    #     # Set trailing stop parameters (adjust trailing_offset as needed)
    #     trailing_trigger = 0.5 * stake  # can be fine-tuned further
    #     trailing_offset = 0.1 * stake

    #     current_trailing_stop = None
    #     self.logger.info(f"Started monitoring contract {contract_id} for stop-loss/trailing stop conditions.")
    #     while True:
    #         req = {"proposal_open_contract": contract_id}
    #         result = await self.api.send(req)
    #         if "proposal_open_contract" not in result:
    #             self.logger.info(f"Contract {contract_id} no longer available for monitoring.")
    #             break
    #         current_profit = float(result["proposal_open_contract"].get("profit", 0.0))
    #         # Check stop-loss
    #         if current_profit <= stop_loss_threshold:
    #             self.logger.info(f"Stop-loss triggered for contract {contract_id} at profit {current_profit:.2f}. Attempting early sell.")
    #             sell_req = {"sell": contract_id, "price": stake}
    #             sell_response = await self.api.sell(sell_req)
    #             self.logger.info(f"Early sell response: {sell_response}")
    #             break
    #         # Check trailing stop conditions if profit is above trigger
    #         if current_profit >= trailing_trigger:
    #             if current_trailing_stop is None or (current_profit - trailing_offset) > current_trailing_stop:
    #                 current_trailing_stop = current_profit - trailing_offset
    #                 self.logger.info(f"Updated trailing stop for contract {contract_id} to {current_trailing_stop:.2f}")
    #             elif current_profit < current_trailing_stop:
    #                 self.logger.info(f"Trailing stop triggered for contract {contract_id} at profit {current_profit:.2f}. Attempting early sell.")
    #                 sell_req = {"sell": contract_id, "price": stake}
    #                 sell_response = await self.api.sell(sell_req)
    #                 self.logger.info(f"Early sell response: {sell_response}")
    #                 break
    #         await asyncio.sleep(10)

    async def check_contract_result(self, contract_id):
        await asyncio.sleep(self.contract_duration * 60 + 10)
        req = {"proposal_open_contract": contract_id}
        result = await self.api.send(req)
        if "proposal_open_contract" in result:
            trade_profit = float(
                result["proposal_open_contract"].get("profit", 0.0))
            self.logger.info(
                f"Contract {contract_id} settled. Profit: {trade_profit:.2f} USDT")
            self.cumulative_pnl += trade_profit
            if trade_profit < 0:
                self.consecutive_loss_count += 1
            else:
                self.consecutive_loss_count = 0
            if self.last_trade_record is not None:
                self.last_trade_record["profit"] = trade_profit
            if self.last_trade_state is not None and self.last_trade_action is not None:
                new_label = self.last_trade_action if trade_profit >= 0 else 1 - self.last_trade_action
                # Boost learning from live experience
            for _ in range(3):  # amplify importance
                self.experience_buffer.append(
                    (self.last_trade_state, new_label))
            self.last_trade_state = None
            self.last_trade_action = None

    async def trading_loop(self):
        self.loop = asyncio.get_running_loop()
        self.connect_api()
        await self.authorize_api()

        # Start background training
        asyncio.create_task(self.training_loop())

        # Wait for training warmup
        while self.training_iterations < self.MIN_TRAINING_CYCLES:
            self.logger.info(f"Waiting for model warmup ({self.training_iterations}/{self.MIN_TRAINING_CYCLES})...")
            await asyncio.sleep(30)

    # Proceed with trading cycles
        while True:
            if self.training_iterations < self.MIN_TRAINING_CYCLES:
                self.logger.info(
                    f"Model warming up (iterations: {self.training_iterations}).")
                await asyncio.sleep(60)
                continue
            if self.consecutive_loss_count >= 10:
                self.logger.warning(
                    "Consecutive loss count exceeded threshold (>=10). Pausing trading for 1 hour.")
                await asyncio.sleep(3600)
                self.consecutive_loss_count = 0
                continue
            if not globals().get("trading_enabled", False):
                self.logger.info("Trading disabled; idling...")
                await asyncio.sleep(1)
                continue
            self.logger.info(
                f"Starting trade cycle #{self.cycle_count + 1} on {self.training_symbol}...")
            try:
                await self.evaluate_and_trade()
            except Exception as e:
                self.logger.error(f"Error during trade cycle: {e}")
            self.logger.info("Trade cycle complete.")
            wait_time = 60  # Default to 1 minute

            try:
                df_latest = await self.fetch_historical_data(count=100, granularity=60)
                if df_latest is not None and not df_latest.empty:
                    df_latest = self.preprocess_data(df_latest)
                    if df_latest is not None and not df_latest.empty:
                        price = float(df_latest.iloc[-1]["close"])
                        atr = float(df_latest.iloc[-1]["atr"])
                        vol_ratio = atr / price if price else 0.001
                        # Clamp for sanity
                        vol_ratio = max(0.001, min(vol_ratio, 0.03))

                        # Scale wait time between 1 min and 15 min
                        wait_time = int(60 + (vol_ratio - 0.001) /
                                        (0.03 - 0.001) * (900 - 60))

                        # OPTIONAL: Adjust wait time based on confidence
                        last_confidence = None
                        if self.trade_history and "confidence" in self.trade_history[-1]:
                            last_confidence = self.trade_history[-1]["confidence"]
                            if last_confidence is not None and last_confidence >= 0.8:
                                # Cut wait in half
                                wait_time = max(30, int(wait_time * 0.5))

                        # OPTIONAL: Add time-of-day throttle (UTC)
                        now_utc_hour = datetime.now(timezone.utc).hour
                        if now_utc_hour < 5 or now_utc_hour > 21:
                            # Minimum 5 mins at night
                            wait_time = max(wait_time, 300)

                        self.logger.info(
                            f"Volatility ratio: {vol_ratio:.5f}, Wait time: {wait_time/60:.2f} mins"
                            + (f", Confidence override: {last_confidence:.2f}" if last_confidence else "")
                            + f", UTC Hour: {now_utc_hour}"
                        )
            except Exception as e:
                self.logger.warning(
                    f"Error determining dynamic wait time: {e}")
                wait_time = 60  # Fallback

            await asyncio.sleep(wait_time)

    async def authorize_api(self):
        try:
            auth_msg = {"authorize": self.api_token}
            auth_response = await self.api.send(auth_msg)
            if auth_response.get("error"):
                self.logger.error(
                    f"Authorization failed: {auth_response['error']}")
                raise Exception("Authorization failed")
            self.logger.info("API authorized successfully.")
            await self.update_balance()
        except Exception as e:
            self.logger.error(f"Error during API authorization: {e}")
            raise

    async def update_balance(self):
        try:
            balance_response = await self.api.send({"balance": 1})
            if "balance" in balance_response:
                new_balance = float(balance_response["balance"]["balance"])
                self.capital = new_balance
                self.fixed_stake = self.capital * 0.01
                self.logger.info(
                    f"Updated balance: {self.capital:.2f} USDT; Fixed stake: {self.fixed_stake:.2f} USDT")
            else:
                self.logger.warning(
                    "Balance information not found in response.")
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")
            self.capital = 18.70
            self.fixed_stake = self.capital * 0.01

    async def fetch_historical_data(self, count=100, granularity=60):
        try:
            if self.training_symbol.upper() in ["BOOM1000", "CRASH1000"]:
                req = {
                    "ticks_history": self.training_symbol.upper(),
                    "count": count,
                    "end": "latest",
                    "granularity": granularity
                }
                response = await self.api.send(req)
                if ("history" in response and
                    "prices" in response["history"] and
                        "times" in response["history"]):
                    df = pd.DataFrame({
                        "datetime": pd.to_datetime(response["history"]["times"], unit="s", utc=True),
                        "price": response["history"]["prices"]
                    })
                    df.set_index("datetime", inplace=True)
                    return df
                else:
                    raise Exception("History data not found in API response")
            else:
                symbol = self.training_symbol
                if symbol.lower().startswith("frx"):
                    symbol = symbol[3:] + "=X"
                df = yf.download(symbol, period="1d",
                                 interval="1m", auto_adjust=True)
                return df
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None

    def connect_api(self):
        try:
            self.logger.info("Connecting to Deriv API...")
            self.api = DerivAPI(api_token=self.api_token, app_id=self.app_id)
        except Exception as e:
            self.logger.error(f"Error connecting to API: {e}")
            raise

    def backtest(self, historical_data):
        """
        A basic backtesting routine that preprocesses historical data, runs predictions,
        and calculates hypothetical PnL. (This is for offline evaluation.)
        """
        df = self.preprocess_data(historical_data.copy())
        if df is None or df.empty:
            self.logger.error(
                "Backtesting data empty or error in preprocessing.")
            return None
        X = df[self.feature_names]
        y_true = df['direction']
        X_scaled = RobustScaler().fit_transform(X)
        X_transformed = self.transform_features(X_scaled)
        try:
            y_pred = self.ml_model.predict(X_transformed)
            pnl = np.where(y_pred == y_true, 1, -1)
            cumulative_pnl = np.cumsum(pnl)
            self.logger.info(
                f"Backtest complete. Final cumulative PnL: {cumulative_pnl[-1]}")
            return cumulative_pnl
        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            return None


# Main Execution: Start Flask and Trading Bot
if __name__ == "__main__":
    bot_instance = DerivTradingBot(
        app_id, api_token, training_symbol="frxEURUSD", contract_duration=30)

    def run_flask():
        app.run(host="0.0.0.0", port=5000, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    async def main():
        await bot_instance.trading_loop()

    asyncio.run(main())
