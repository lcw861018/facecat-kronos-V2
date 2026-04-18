# -*- coding:UTF-8 -*-
#! python3
from facecat import *
#這里可能需要pip install requests
import requests
from requests.adapters import HTTPAdapter
import random
from datetime import datetime
import json
import os
from stock import *


import pandas as pd
import sys
from model import Kronos, KronosTokenizer, KronosPredictor
import torch
from huggingface_hub import snapshot_download

latestDataStr = ""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACECAT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TOKENIZER_DIR = os.path.join(BASE_DIR, "model", "Kronos-Tokenizer-base")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "model", "Kronos-small")
LOCAL_TOKENIZER_DIR_ALT = os.path.join(FACECAT_DIR, "model", "Kronos-Tokenizer-base")
LOCAL_MODEL_DIR_ALT = os.path.join(FACECAT_DIR, "model", "Kronos-small")
HF_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
HF_MODEL_REPO = "NeoQuasar/Kronos-small"

def has_pretrained_weights(model_dir):
	"""Only treat a folder as ready when config and weights both exist."""
	if not os.path.isdir(model_dir):
		return False
	if not os.path.isfile(os.path.join(model_dir, "config.json")):
		return False
	weight_patterns = (".safetensors", ".bin", ".pt", ".pth")
	for file_name in os.listdir(model_dir):
		if file_name.endswith(weight_patterns):
			return True
	return False


def get_candidate_model_dirs(local_dir):
	candidates = [local_dir]
	if "Tokenizer" in local_dir:
		candidates.append(LOCAL_TOKENIZER_DIR_ALT)
	elif "Kronos-small" in local_dir:
		candidates.append(LOCAL_MODEL_DIR_ALT)
	ordered = []
	for candidate in candidates:
		if candidate not in ordered:
			ordered.append(candidate)
	return ordered


def ensure_local_pretrained_dir(local_dir, repo_id):
	"""Download the official snapshot when the bundled folder is incomplete."""
	candidates = get_candidate_model_dirs(local_dir)
	for candidate in candidates:
		if has_pretrained_weights(candidate):
			return candidate
	target_dir = candidates[-1]
	os.makedirs(target_dir, exist_ok=True)
	print(f"[startup] downloading missing model assets: {repo_id} -> {target_dir}")
	snapshot_download(repo_id=repo_id, local_dir=target_dir)
	if has_pretrained_weights(target_dir):
		return target_dir
	raise FileNotFoundError(f"{repo_id} downloaded, but no weight file was found in {target_dir}")
# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base",force_download=True, cache_dir="model/Kronos-Tokenizer-base")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-small",force_download=True, cache_dir="model/Kronos-small")
if torch.cuda.is_available():
	device = "cuda:0"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"
tokenizer = None
model = None
predictor = None
print(f"[startup] app ready on {device}")
latestDataStr = ""
findMyCharts = []
charts = []
currentCode = "2330"
clientTicks = dict()
priceRowMap = dict()
latestQuote = {}

TAIWAN_WATCHLIST = [
	{"code": "2330", "name": "台積電", "market": "tse", "suffix": "TW"},
	{"code": "2317", "name": "鴻海", "market": "tse", "suffix": "TW"},
	{"code": "2454", "name": "聯發科", "market": "tse", "suffix": "TW"},
	{"code": "2308", "name": "台達電", "market": "tse", "suffix": "TW"},
	{"code": "2603", "name": "長榮", "market": "tse", "suffix": "TW"},
	{"code": "2881", "name": "富邦金", "market": "tse", "suffix": "TW"},
	{"code": "2882", "name": "國泰金", "market": "tse", "suffix": "TW"},
	{"code": "0050", "name": "元大台灣50", "market": "tse", "suffix": "TW"},
	{"code": "00878", "name": "國泰永續高股息", "market": "tse", "suffix": "TW"},
	{"code": "6488", "name": "環球晶", "market": "otc", "suffix": "TWO"},
]
TAIWAN_STOCK_MAP = {item["code"]: item for item in TAIWAN_WATCHLIST}
HTTP_HEADERS = {
	"User-Agent": "Mozilla/5.0",
	"Accept": "application/json,text/plain,*/*",
}


def create_stock_profile(code, name, category, market="tse", suffix="TW"):
	return {
		"code": code,
		"name": name,
		"category": category,
		"market": market,
		"suffix": suffix,
	}


TAIWAN_STOCK_GROUPS = [
	("半導體", [
		create_stock_profile("2330", "台積電", "半導體"),
		create_stock_profile("2454", "聯發科", "半導體"),
		create_stock_profile("3034", "聯詠", "半導體"),
		create_stock_profile("3711", "日月光投控", "半導體"),
		create_stock_profile("2379", "瑞昱", "半導體"),
		create_stock_profile("2327", "國巨", "半導體"),
		create_stock_profile("6488", "環球晶", "半導體", market="otc", suffix="TWO"),
		create_stock_profile("5347", "世界", "半導體", market="otc", suffix="TWO"),
		create_stock_profile("8299", "群聯", "半導體", market="otc", suffix="TWO"),
	]),
	("AI伺服器", [
		create_stock_profile("2317", "鴻海", "AI伺服器"),
		create_stock_profile("2382", "廣達", "AI伺服器"),
		create_stock_profile("3231", "緯創", "AI伺服器"),
		create_stock_profile("6669", "緯穎", "AI伺服器"),
		create_stock_profile("2356", "英業達", "AI伺服器"),
		create_stock_profile("2308", "台達電", "AI伺服器"),
		create_stock_profile("2301", "光寶科", "AI伺服器"),
		create_stock_profile("3017", "奇鋐", "AI伺服器"),
	]),
	("電子零組件", [
		create_stock_profile("2357", "華碩", "電子零組件"),
		create_stock_profile("2345", "智邦", "電子零組件"),
		create_stock_profile("3008", "大立光", "電子零組件"),
		create_stock_profile("8069", "元太", "電子零組件", market="otc", suffix="TWO"),
	]),
	("航運", [
		create_stock_profile("2603", "長榮", "航運"),
		create_stock_profile("2609", "陽明", "航運"),
		create_stock_profile("2615", "萬海", "航運"),
	]),
	("金融", [
		create_stock_profile("2881", "富邦金", "金融"),
		create_stock_profile("2882", "國泰金", "金融"),
		create_stock_profile("2884", "玉山金", "金融"),
		create_stock_profile("2885", "元大金", "金融"),
		create_stock_profile("2886", "兆豐金", "金融"),
		create_stock_profile("2891", "中信金", "金融"),
		create_stock_profile("5871", "中租-KY", "金融"),
	]),
	("電信", [
		create_stock_profile("2412", "中華電", "電信"),
		create_stock_profile("3045", "台灣大", "電信"),
		create_stock_profile("4904", "遠傳", "電信"),
	]),
	("傳產", [
		create_stock_profile("1216", "統一", "傳產"),
		create_stock_profile("1301", "台塑", "傳產"),
		create_stock_profile("1303", "南亞", "傳產"),
		create_stock_profile("1326", "台化", "傳產"),
		create_stock_profile("2002", "中鋼", "傳產"),
		create_stock_profile("2207", "和泰車", "傳產"),
	]),
	("ETF", [
		create_stock_profile("0050", "元大台灣50", "ETF"),
		create_stock_profile("0056", "元大高股息", "ETF"),
		create_stock_profile("00878", "國泰永續高股息", "ETF"),
		create_stock_profile("00919", "群益台灣精選高息", "ETF"),
		create_stock_profile("00929", "復華台灣科技優息", "ETF"),
	]),
]
TAIWAN_WATCHLIST = []
for _, stocks in TAIWAN_STOCK_GROUPS:
	TAIWAN_WATCHLIST.extend(stocks)
TAIWAN_STOCK_MAP = {item["code"]: item for item in TAIWAN_WATCHLIST}


def safe_float(value, default=0.0):
	"""Convert loose number strings from public APIs into floats."""
	if value in (None, "", "-", "--", "---", "null"):
		return default
	if isinstance(value, (int, float)):
		return float(value)
	try:
		return float(str(value).replace(",", ""))
	except (TypeError, ValueError):
		return default


def get_stock_profile(code):
	"""Look up Taiwan stock metadata by 4/5 digit code."""
	normalized = str(code).split(".")[0]
	return TAIWAN_STOCK_MAP.get(normalized, {
		"code": normalized,
		"name": normalized,
		"category": "其他",
		"market": "tse",
		"suffix": "TW",
	})


def get_yahoo_symbol(code):
	profile = get_stock_profile(code)
	return f"{profile['code']}.{profile['suffix']}"


def get_mis_channel(code):
	profile = get_stock_profile(code)
	return f"{profile['market']}_{profile['code']}.tw"


def parse_twse_order_book(price_text, volume_text, reverse=False):
	"""Parse MIS order book strings into [(price, volume), ...]."""
	prices = [safe_float(item) for item in str(price_text or "").split("_")]
	volumes = [safe_float(item) for item in str(volume_text or "").split("_")]
	book = []
	for price, volume in zip(prices, volumes):
		if price > 0:
			book.append((price, volume))
	if reverse:
		book.reverse()
	return book


def build_latest_quote(stock_info):
	"""Convert MIS quote payload into the structure used by the detail panel."""
	code = stock_info.get("c", "")
	profile = get_stock_profile(code)
	close = safe_float(stock_info.get("z")) or safe_float(stock_info.get("b").split("_")[0] if stock_info.get("b") else 0)
	last_close = safe_float(stock_info.get("y"))
	open_price = safe_float(stock_info.get("o"))
	high = safe_float(stock_info.get("h"))
	low = safe_float(stock_info.get("l"))
	volume = safe_float(stock_info.get("v"))
	total_volume = safe_float(stock_info.get("tv"))
	if total_volume <= 0:
		total_volume = volume
	amount = close * total_volume
	sell_book = parse_twse_order_book(stock_info.get("a"), stock_info.get("f"), reverse=True)
	buy_book = parse_twse_order_book(stock_info.get("b"), stock_info.get("g"))
	return {
		"code": code,
		"name": profile["name"],
		"category": profile["category"],
		"close": close,
		"high": high,
		"low": low,
		"open": open_price,
		"volume": total_volume,
		"amount": amount,
		"last_close": last_close,
		"sell_book": sell_book[:5],
		"buy_book": buy_book[:5],
		"time": stock_info.get("tlong") or stock_info.get("tt"),
	}


def build_mis_url(codes):
	"""Build a TWSE MIS quote URL for one or more stock codes."""
	ex_ch = "|".join([get_mis_channel(code) for code in codes])
	return f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={ex_ch}&json=1&delay=0"


def build_yahoo_chart_url(code, interval, range_value):
	"""Build a Yahoo Finance chart endpoint for Taiwan stocks."""
	symbol = get_yahoo_symbol(code)
	return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_value}&includePrePost=false&events=div%2Csplits"


def parse_yahoo_history(result_text):
	"""Parse Yahoo chart JSON into SecurityData records."""
	data = json.loads(result_text)
	result = data.get("chart", {}).get("result", [])
	if not result:
		return []
	chart_result = result[0]
	timestamps = chart_result.get("timestamp") or []
	quote_list = chart_result.get("indicators", {}).get("quote", [])
	if not quote_list:
		return []
	quote = quote_list[0]
	opens = quote.get("open", [])
	highs = quote.get("high", [])
	lows = quote.get("low", [])
	closes = quote.get("close", [])
	volumes = quote.get("volume", [])
	data_list = []
	for index, ts in enumerate(timestamps):
		close = safe_float(closes[index] if index < len(closes) else 0)
		open_price = safe_float(opens[index] if index < len(opens) else close, close)
		high = safe_float(highs[index] if index < len(highs) else close, close)
		low = safe_float(lows[index] if index < len(lows) else close, close)
		volume = safe_float(volumes[index] if index < len(volumes) else 0)
		if close <= 0:
			continue
		stock_data = SecurityData()
		stock_data.date = int(ts)
		stock_data.open = open_price
		stock_data.high = high if high > 0 else close
		stock_data.low = low if low > 0 else close
		stock_data.close = close
		stock_data.volume = volume
		stock_data.amount = close * volume * 1000
		data_list.append(stock_data)
	return data_list

def transToPanda(datas):
	data_list = []
	for data in datas:
		data_list.append({
			'timestamps': datetime.fromtimestamp(data.date),
			'open': data.open,
			'high': data.high,
			'low': data.low,
			'close': data.close,
			'volume': data.volume,
			'amount': data.amount
		})
	df = pd.DataFrame(data_list)
	return df

def transToChartData(pred_df):
    data_list = []
    for index, row in pred_df.iterrows():
        data = SecurityData()
        data.open = row['open']
        data.date = time.mktime(index.timetuple())
        data.high = row['high']
        data.low = row['low']
        data.close = row['close']
        data.volume = row['volume']
        data.amount = row['amount']
        data_list.append(data)
    return data_list

def progress_callback(progress, total):
	progressDiv = gPaint.findView("progress")
	progressDiv.text = str(progress/total)
	progressDiv.invalidate()


def ensure_predictor_loaded():
	"""Load predictor lazily so the stock UI can start without model files."""
	global tokenizer
	global model
	global predictor
	if predictor is not None:
		return predictor
	try:
		tokenizer = KronosTokenizer.from_pretrained(
			ensure_local_pretrained_dir(LOCAL_TOKENIZER_DIR, HF_TOKENIZER_REPO)
		)
		model = Kronos.from_pretrained(
			ensure_local_pretrained_dir(LOCAL_MODEL_DIR, HF_MODEL_REPO)
		)
		predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
		print(f"[startup] model ready on {device}")
		return predictor
	except Exception as ex:
		popupWindow(f"AI 模型載入失敗: {ex}")
		return None

def predict(chart):
	"""
	使用歷史K線數據對未來進行預測
	"""
	preButton = gPaint.findView("preButton")
	preButton.enabled = False
	progressDiv = gPaint.findView("progress")
	progressDiv.text = "-1"
	progressDiv.invalidate()
	preButton.invalidate()
	lookback, pred_len, adjust_message = normalize_prediction_params(chart)
	mode = chart.preMode
	
	if lookback is None or (len(chart.datas) < lookback and mode == "predict"):
		popupWindow(f"k線數量{len(chart.datas)}小於訓練數量{lookback}")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	elif len(chart.datas) < lookback + pred_len and mode == "backtest":
		popupWindow(f"k線數量{len(chart.datas)}小於訓練數量{lookback + pred_len}")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	if mode == "predict": # 預測模式
		print(f"Using device: {device} mode: {mode}")
		# 可見k線數量
		klines = chart.lastVisibleIndex - chart.firstVisibleIndex 
		if pred_len > 50 and klines > 50:
			chart.firstVisibleIndex += 50
		elif pred_len <= 50 and klines > pred_len:
			chart.firstVisibleIndex += pred_len
		chart.invalidate()
	# 獲取圖表數據，轉化成panda格式
		df = transToPanda(chart.datas)
		df['timestamps'] = pd.to_datetime(df['timestamps'])

		df = df.sort_values('timestamps').reset_index(drop=True)

		x_df = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
		x_timestamp = df.tail(lookback)['timestamps'].reset_index(drop=True)

		# 3. 生成未來的時間戳 (這里我們假設是連續的未來日期)
		last_timestamp = x_timestamp.iloc[-1]
		# 注意：對於股票市場，應該生成交易日，但為簡單起見，我們先生成連續日歷日
		y_timestamp = pd.Series(pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=pred_len))
		pred_df = predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		chart.datas2 = transToChartData(pred_df)
	elif mode == "backtest": # 回測模式
		# popupWindow()
		print(f"Using device: {device} mode: {mode}")
		df = transToPanda(chart.datas)
		df['timestamps'] = pd.to_datetime(df['timestamps'])

		df = df.sort_values('timestamps').reset_index(drop=True)

		test_df = df.tail(lookback + pred_len).reset_index(drop=True)

		x_df = test_df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
		x_timestamp = test_df.loc[:lookback-1, 'timestamps']
		y_timestamp = test_df.loc[lookback:lookback+pred_len-1, 'timestamps']

		# 4. Make Prediction
		pred_df = predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		print(pred_df)
		
		chart.datas2 = transToChartData(pred_df)
	preButton.enabled = True
	preButton.invalidate()

def predict_in_thread(chart):
	"""
	在獨立線程中運行預測，以避免UI阻塞.
	"""
	predict(chart)
	chart.invalidate()

def startHttpRequest(url, callBack, tag):
	"""開始Http請求
	url:地址
	callBack回調"""
	data = FCData()
	data.key = url
	data.callBack = callBack
	try:
		s = requests.Session()
		s.mount('http://', HTTPAdapter(max_retries=3))
		s.mount('https://', HTTPAdapter(max_retries=3))
		response = s.get(url, headers=HTTP_HEADERS, timeout=15)
		result = response.text
		data.success = True
		data.data = result
	except requests.exceptions.RequestException as e:
		data.success = False
		data.data = str(e)
	data.tag = tag
	gPaint.addData(data)
	user32.PostMessageW(gPaint.hWnd, 0x0401, 0, 0)

def startQueryNewData():
	"""開始請求最新數據"""
	while gPaint.hWnd != 0:
		time.sleep(3)
		queryNewData()

def startQueryPriceData():
	"""開始請求最新數據"""
	while gPaint.hWnd != 0:
		time.sleep(6)
		queryPrice()

def httpRequest(url, callBack, tag):
	"""進行Http請求
	url:地址
	callBack回調"""
	thread = threading.Thread(target=startHttpRequest, args=(url, callBack, tag))
	thread.start()

def onPaint(view, paint, clipRect):
	"""繪制視圖
	view:視圖
	paint:繪圖對象
	clipRect:區域"""
	if view.viewType == "latestdiv":
		drawLatestDiv(view, paint, clipRect)
	else:
		onPaintDefault(view, paint, clipRect)


def drawMyDiv(view, paint, clipRect):
		# elif view.viewName == "temperatureDiv" or view.viewName == "topPDiv":
	tSize = paint.textSize(view.text, "Default,14")
	paint.drawText(view.text, "rgb(255,255,255)",  "Default,14", 3, (view.size.cy - tSize.cy)/2)
	paint.drawLine("rgb(255,255,255)", 1, 0, view.size.cx-1, 0, view.size.cx-1, view.size.cy)
	
def drawProgressDiv(view, paint, clipRect):
	if view.text == "-1":
		paint.drawText("訓練中...", "rgb(255,255,255)", "Default,14", 3, 7)
	else:
		cx = float(view.text) * view.size.cx
		if cx > 0 and cx < view.size.cx:
			paint.fillRect(view.textColor, 0, 0, cx, view.size.cy)

def drawPreChart(view, paint, clipRect):
	drawChartStock(view, paint, clipRect)
	if view.datas2 != None and len(view.datas2) > 0:
		cWidth = int(view.hScalePixel - 3) / 2
		splitIndex = view.lastVisibleIndex + 1
		if view.preMode == "backtest":
			splitIndex = view.lastVisibleIndex - view.pred_len + 1
			if view.lastVisibleIndex + 1 < len(view.datas):
				splitIndex = len(view.datas) - 1 - view.pred_len + 1
		elif view.preMode == "predict":
			if view.lastVisibleIndex + 1 < len(view.datas):
				return
		splitX = getChartX(view, splitIndex) - cWidth
		paint.drawLine("rgb(150,150,150)", 1, 0, splitX, 0, splitX, view.size.cy)
		for i in range(0,len(view.datas2)):
			index = view.lastVisibleIndex + i + 1
			
			if view.preMode == "backtest":
				index = view.lastVisibleIndex - view.pred_len + i + 1
				if view.lastVisibleIndex + 1 < len(view.datas):
					index = len(view.datas) - view.pred_len + i + 1
			x = getChartX(view, index)
			if x > view.size.cx - view.rightVScaleWidth:
				break
			m_open = view.datas2[i].open
			m_close = view.datas2[i].close
			m_high = view.datas2[i].high
			m_low = view.datas2[i].low
			openY = getChartY(view, 0, m_open)
			closeY = getChartY(view, 0, m_close)
			highY = getChartY(view, 0, m_high)
			lowY = getChartY(view, 0, m_low)
			if m_close >= m_open:				
				paint.fillRect(view.upColor2, x, highY, x + view.lineWidth, lowY)
				if cWidth > 0:
					if int(closeY) == int(openY):
						paint.drawLine(view.upColor2, 1, 0, x - cWidth, closeY, x + cWidth, closeY)
					else:
						paint.fillRect(view.upColor2, x - cWidth, closeY, x + cWidth + 1, openY)
			else:
				paint.fillRect(view.downColor2, x, highY, x + view.lineWidth, lowY)
				if cWidth > 0:
					paint.fillRect(view.downColor2, x - cWidth, openY, x + cWidth + 1, closeY)
			volY = getChartY(view, 1, view.datas2[i].volume)
			zeroY = getChartY(view, 1, 0)
			if m_close >= m_open:
				barColor = view.upColor2
				if view.volColor != "none":
					barColor = view.volColor

				if cWidth > 0:
					paint.fillRect(barColor, x - cWidth, volY, x + cWidth + 1, zeroY)
				else:
					paint.drawLine(barColor, view.lineWidth, 0, x - cWidth, volY, x + cWidth, zeroY)
			else:
				barColor = view.downColor2
				if view.volColor != "none":
					barColor = view.volColor
				if cWidth > 0:
					paint.fillRect(barColor, x - cWidth, volY, x + cWidth + 1, zeroY)
				else:
					paint.drawLine(barColor, view.lineWidth, 0, x - cWidth, volY, x + cWidth, zeroY)

def drawUpButton(button, paint, clipRect):
	"""繪制溫度按鈕
	view:視圖
	paint:繪圖對象
	clipRect:區域"""
	"""重繪按鈕 
	button:視圖 
	paint:繪圖對象 
	clipRect:裁剪區域"""
	r_left = 0
	r_right = button.size.cx
	r_top = 0
	r_bottom = button.size.cy
	r_width = r_right - r_left
	#常規情況
	if button.backColor != "none":
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.textColor, apt)
	#鼠標按下
	if button == paint.touchDownView:
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.pushedColor, apt)
	#鼠標懸停
	elif button == paint.touchMoveView:
		apt = []
		apt.append(FCPoint(r_left + r_width/2,r_top))
		apt.append(FCPoint(r_left, r_top + r_width/2))
		apt.append(FCPoint(r_right, r_top + r_width/2))
		paint.fillPolygon(button.hoveredColor, apt)

def drawDownButton(button, paint, clipRect):
	r_left = 0
	r_right = button.size.cx
	r_top = 0
	r_bottom = button.size.cy
	r_width = r_right - r_left
	#常規情況
	if button.backColor != "none":
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.textColor, apt2)
	#鼠標按下
	if button == paint.touchDownView:
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.pushedColor, apt2)
	#鼠標懸停
	elif button == paint.touchMoveView:
		apt2 = []
		apt2.append(FCPoint(r_left + r_width/2,r_bottom))
		apt2.append(FCPoint(r_right, r_bottom - r_width/2))
		apt2.append(FCPoint(r_left, r_bottom - r_width/2))
		paint.fillPolygon(button.hoveredColor, apt2)

def drawLatestDiv(view, paint, clipRect):
	"""繪制買賣檔
	view:視圖
	paint:繪圖對象
	clipRect:區域"""
	global latestDataStr
	avgHeight = 20
	drawFont = "Default,14"
	textColor = "rgb(175,196,228)"
	if view.paint.defaultUIStyle == "light":
		textColor = "rgb(0,0,0)"
	dTop = 30
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	dataStrs = latestDataStr.split(",")
	lastClose = 0
	priceList = []
	volList = []
	buySellTexts = []
	if len(dataStrs) > 10:
		paint.drawText(dataStrs[0], textColor, "Default,14", 5, 7)
		paint.drawText(dataStrs[1], "rgb(194,151,18)", "Default,14", 80, 7)
		lastClose = float(dataStrs[8])
		priceList.append(float(dataStrs[23]))
		priceList.append(float(dataStrs[22]))
		priceList.append(float(dataStrs[21]))
		priceList.append(float(dataStrs[20]))
		priceList.append(float(dataStrs[19]))
		priceList.append(float(dataStrs[9]))
		priceList.append(float(dataStrs[10]))
		priceList.append(float(dataStrs[11]))
		priceList.append(float(dataStrs[12]))
		priceList.append(float(dataStrs[13]))

		volList.append(float(dataStrs[28]))
		volList.append(float(dataStrs[27]))
		volList.append(float(dataStrs[26]))
		volList.append(float(dataStrs[25]))
		volList.append(float(dataStrs[24]))
		volList.append(float(dataStrs[14]))
		volList.append(float(dataStrs[15]))
		volList.append(float(dataStrs[16]))
		volList.append(float(dataStrs[17]))
		volList.append(float(dataStrs[18]))

	buySellTexts.append("賣5")
	buySellTexts.append("賣4")
	buySellTexts.append("賣3")
	buySellTexts.append("賣2")
	buySellTexts.append("賣1")
	buySellTexts.append("買1")
	buySellTexts.append("買2")
	buySellTexts.append("買3")
	buySellTexts.append("買4")
	buySellTexts.append("買5")
	maxVol = maxValue(volList)
	for i in range(0, 10):
		tSize = paint.textSize(buySellTexts[i], drawFont)
		paint.drawText(buySellTexts[i], textColor, drawFont, 5, dTop + avgHeight / 2 - tSize.cy / 2)
		if len(priceList) > 0:
			price = priceList[i]
			upDownColor = "rgb(255,82,82)"
			upDownColor2 = "rgb(50,0,0)"
			if price < lastClose:
				upDownColor = "rgb(46,255,50)"
				upDownColor2 = "rgb(0,50,0)"
				if paint.defaultUIStyle == "light":
					upDownColor = "rgb(0,200,0)"
					upDownColor2 = "rgba(0,200,0,50)"
			paint.drawText(toFixed(priceList[i], 2), upDownColor, drawFont, 50, dTop + avgHeight / 2 - tSize.cy / 2)
			volText = toFixed(volList[i] / 100, 0)
			volTextSize = paint.textSize(volText, drawFont)
			paint.drawText(volText, textColor, drawFont, view.size.cx - volTextSize.cx - 10, dTop + avgHeight / 2 - volTextSize.cy / 2)
		dTop += avgHeight
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	paint.drawText("現價", textColor, drawFont, 5, dTop + 10)
	paint.drawText("幅度", textColor, drawFont, 5, dTop + 35)
	paint.drawText("總額", textColor, drawFont, 5, dTop + 60)
	paint.drawText("總量", textColor, drawFont, 5, dTop + 85)
	paint.drawText("開盤", textColor, drawFont, 110, dTop + 10)
	paint.drawText("振幅", textColor, drawFont, 110, dTop + 35)
	paint.drawText("最高", textColor, drawFont, 110, dTop + 60)
	paint.drawText("最低", textColor, drawFont, 110, dTop + 85)
	if len(dataStrs) > 10:
		close = float(dataStrs[2])
		high = float(dataStrs[3])
		low = float(dataStrs[4])
		open = float(dataStrs[5])
		volume = float(dataStrs[6])
		amount = float(dataStrs[7])
		diff = 0
		if lastClose > 0:
			diff = 100 * (close - lastClose) / lastClose
		diff2 = 0
		if lastClose > 0:
			diff2 = 100 * (high - lastClose) / lastClose - 100 * (low - lastClose) / lastClose
		paint.drawText(toFixed(close, 2), getPriceColor(close, lastClose), drawFont, 40, dTop + 10)
		paint.drawText(toFixed(diff, 2) + "%", getPriceColor(close, lastClose), drawFont, 40, dTop + 35)
		paint.drawText(toFixed(amount / 10000, 0), textColor, drawFont, 40, dTop + 60)
		paint.drawText(toFixed(volume / 10000, 0), textColor, drawFont, 40, dTop + 85)

		paint.drawText(toFixed(open, 2), getPriceColor(open, lastClose), drawFont, 150, dTop + 10)
		paint.drawText(toFixed(diff2, 2) + "%", getPriceColor(close, lastClose), drawFont, 150, dTop + 35)
		paint.drawText(toFixed(high, 2), getPriceColor(high, lastClose), drawFont, 150, dTop + 60)
		paint.drawText(toFixed(low, 2), getPriceColor(low, lastClose), drawFont, 150, dTop + 85)

def historyDataCallBack(data):
	"""歷史數據回調"""
	if data.success:
		code = data.tag[0]
		name = data.tag[1]
		cycle = data.tag[2]
		myCharts = data.tag[3]
		result = data.data
		dataList = []
		strs = result.split("\r\n")
		#分時線處理
		for c in range(0, len(myCharts)):
			myChart = myCharts[c]
			chart = myChart.views[0].secondView.views[0]
			myCycle = int(myChart.exAttributes["cycle"])
			if myCycle == 0:
				if cycle == 0:
					fStrs = strs[0].split(" ")
					if len(fStrs) >= 3:
						chart.firstOpen = float(fStrs[2])
					else:
						chart.firstOpen = 0
			else:
				chart.firstOpen = 0
		for i in range(2, len(strs)):
			subStrs = strs[i].split(",")
			if len(subStrs) >= 7:
				data = SecurityData()
				if cycle < 1440:
					dateStr = subStrs[0] + " " + subStrs[1][0:2] + ":" + subStrs[1][2:4] + ":00"
					data.open = float(subStrs[2])
					data.high = float(subStrs[3])
					data.low = float(subStrs[4])
					data.close = float(subStrs[5])
					data.volume = float(subStrs[6])
					date_obj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
					data.date = time.mktime(date_obj.timetuple())
					#分時線處理
					if cycle == 0 and (data.volume > 0 or len(dataList) == 0):
						for c in range(0, len(myCharts)):
							myChart = myCharts[c]
							myCycle = int(myChart.exAttributes["cycle"])
							if myCycle == 0:
								chart = myChart.views[0].secondView.views[0]
								chart.lastValidIndex = len(dataList)
								if chart.firstOpen == 0:
									chart.firstOpen = data.close
				else:
					data.open = float(subStrs[1])
					data.high = float(subStrs[2])
					data.low = float(subStrs[3])
					data.close = float(subStrs[4])
					data.volume = float(subStrs[5])
					dateStr = subStrs[0]
					date_obj = datetime.strptime(dateStr, "%Y-%m-%d")
					data.date = time.mktime(date_obj.timetuple())
				dataList.append(data)
		for c in range(0, len(myCharts)):
			myChart = myCharts[c]
			myCycle = int(myChart.exAttributes["cycle"])
			chart = myChart.views[0].secondView.views[0]
			copyDatas = []
			for d in range(0, len(dataList)):
				copyDatas.append(dataList[d])
			#分時線
			if cycle == 0 and myCycle == 0:
				chart.autoFillHScale = True
				chart.cycle = "trend"
			#分鐘線
			elif cycle == 1 and (myCycle > 0 and myCycle < 1440):
				chart.cycle = "minute"
				if myCycle > 1:
					newDatas = []
					multiMinuteSecurityDatas(newDatas, copyDatas, myCycle)
					copyDatas = newDatas
			#日線
			elif cycle == 1440 and myCycle >= 1440:
				chart.cycle = "day"
				if myCycle == 10080:
					newDatas = []
					getHistoryWeekDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 43200:
					newDatas = []
					getHistoryMonthDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 129600:
					newDatas = []
					getHistorySeasonDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 259200:
					newDatas = []
					getHistoryHalfYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 518400:
					newDatas = []
					getHistoryYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
			#不符合的K線
			else:
				continue
			clientTicks[c] = ClientTickDataCache()
			setChartTitle(chart, code, name, myCycle)
			chart.lastVisibleKey = 0
			chart.firstVisibleIndex = -1
			chart.lastVisibleIndex = -1
			chart.datas = copyDatas
			maxVisibleRecord = getChartMaxVisibleCount(chart, chart.hScalePixel, getChartWorkAreaWidth(chart))
			if maxVisibleRecord > len(chart.datas):
				chart.firstVisibleIndex = 0
			else:
				chart.firstVisibleIndex = chart.lastVisibleIndex - maxVisibleRecord + 1
			resetChartVisibleRecord(chart)
			checkChartLastVisibleIndex(chart)
			calcChartIndicator(chart)
			chart.invalidate()

def queryHistoryData(code, name, cycle, myCharts):
	"""請求歷史數據"""
	url = "http://www.jjmfc.com:9968/quote?func=getkline&code=" + code +  "&cycle=" + str(cycle) + "&count=5000"
	if cycle == 0:
		url = "http://www.jjmfc.com:9968/quote?func=getkline&code=" + code +  "&cycle=0&count=240"
	tag = []
	tag.append(code)
	tag.append(name)
	tag.append(cycle)
	tag.append(myCharts)
	httpRequest(url, historyDataCallBack, tag)

def newDataCallBack(data):
	"""最新數據回調"""
	if data.success:
		global latestDataStr
		result = data.data
		latestDataStr = result
		dataStrs = latestDataStr.split(",")
		if len(dataStrs) > 0:
			code = dataStrs[0]
			close = float(dataStrs[2])
			volume = float(dataStrs[6])
			amount = float(dataStrs[7])
			dateStr = dataStrs[29].replace("\r\n", "")
			date_obj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
			date = time.mktime(date_obj.timetuple())
			for c in range(0, len(findMyCharts)):
				myChart = findMyCharts[c]
				chart = charts[c]
				if chart.viewName=="preChart":
					return
				if code == chart.text.split(" ")[0]:
					if len(chart.datas) > 0:
						myCycle = int(myChart.exAttributes["cycle"])
						latestData = SecurityData()
						latestData.close = close
						latestData.open = latestData.close
						latestData.high = latestData.close
						latestData.low = latestData.close
						latestData.volume = volume
						latestData.amount = amount
						latestData.date = date
						cTick = clientTicks[c]
						if len(cTick.code) == 0:
							cTick.code = currentCode
							cTick.lastAmount = amount
							cTick.lastVolume = volume
							cTick.lastDate = latestData.date
						oldDataSize = len(chart.datas)
						if myCycle == 0:
							for d in range(0, len(chart.datas)):
								if chart.datas[d].volume > 0:
									chart.lastValidIndex = d
						if chart.lastRecordIsVisible:
							newDataSize = len(chart.datas)
							if newDataSize > oldDataSize :
								chart.firstVisibleIndex = chart.firstVisibleIndex + 1
								chart.lastVisibleIndex = chart.lastVisibleIndex + 1
					
						checkChartLastVisibleIndex(chart)
						calcChartIndicator(chart)
		gPaint.update()

def queryNewData():
	"""請求最新數據"""
	url = "http://www.jjmfc.com:9968/quote?func=getnewdata&codes=" + currentCode
	tag = []
	httpRequest(url, newDataCallBack, tag)

def setChartTheme(chart, index):
	"""黑色風格"""
	if chart.paint.defaultUIStyle == "dark":
		chart.backColor = "rgb(0,0,0)"
		chart.borderColor = "none"
		chart.textColor = "rgb(175,196,228)"
		chart.scaleColor = "rgb(75,75,75)"
		chart.crossTipColor = "rgb(50,50,50)"
		chart.crossLineColor = "rgb(100,100,100)"
		chart.gridColor = "rgb(50,50,50)"
		if index > 0:
			chart.upColor = "rgb(186,56,18)"
			chart.downColor = "rgb(31,182,177)"
		else:
			chart.upColor = "rgb(255,82,82)"
			chart.downColor = "rgb(46,255,50)"
		chart.barStyle = "rect2"
		chart.candleStyle = "rect2"
		chart.trendColor = "rgb(255,255,255)"
		chart.hScaleTextColor = "rgb(194,151,18)"
		chart.vScaleTextColor = "rgb(194,151,18)"
	elif chart.paint.defaultUIStyle == "light":
		chart.backColor = "rgb(255,255,255)"
		chart.borderColor = "none"
		chart.textColor = "rgb(0,0,0)"
		chart.scaleColor = "rgb(175,175,175)"
		chart.crossTipColor = "rgb(200,200,200)"
		chart.crossLineColor = "rgb(150,150,150)"
		chart.gridColor = "rgb(200,200,200)"
		chart.trendColor = "rgb(50,50,50)"

def setChartTitle(chart, code, name, intCycle):
	"""設置圖表的標題"""
	chart.text = code + " " + name
	if intCycle == 0:
		chart.text += " 分時"
	elif intCycle < 1440:
		chart.text += " " + str(intCycle) + "分鐘"
	elif intCycle == 1440:
		chart.text += " 日線"
	elif intCycle == 10080:
		chart.text += " 周線"
	elif intCycle == 43200:
		chart.text += " 月線"
	elif intCycle == 129600:
		chart.text += " 季線"
	elif intCycle == 259200:
		chart.text += " 半年線"
	elif intCycle == 518400:
		chart.text += " 年線"

def getPriceColor(price, comparePrice):
	"""獲取價格數據"""
	if gPaint.defaultUIStyle == "dark":
		if price != 0:
			if price > comparePrice:
				return "rgb(255,82,82)"
			elif price < comparePrice:
				return "rgb(46,255,50)"
		return "rgb(190,190,235)"
	else:
		if price != 0:
			if price > comparePrice:
				return "rgb(255,82,82)"
			elif price < comparePrice:
				return "rgb(0,200,0)"
		return "rgb(0,0,0)"

def findViewsByType(findType, views, refViews):
	"""查找同類型視圖"""
	size = len(views)
	for i in range(0, size):
		view = views[i]
		if view.viewType == findType:
			refViews.append(view)
		elif len(view.views) > 0:
			findViewsByType(findType, view.views, refViews)

def onClickGridCell(grid, row, gridColumn, cell, firstTouch, firstPoint, secondTouch, secondPoint, clicks):
	"""點擊單元格"""
	global currentCode
	code = row.cells[1].value
	name = row.cells[2].value
	chart =gPaint.findView("preChart")
	chart.datas2 = []
	queryHistoryData(code, name, 0, findMyCharts)
	queryHistoryData(code, name, 1, findMyCharts)
	queryHistoryData(code, name, 1440, findMyCharts)
	currentCode = code
	queryNewData()
	invalidate(grid.paint)

def onClick(view, firstTouch, firstPoint, secondTouch, secondPoint, clicks):
	"""視圖的鼠標點擊方法
	view 視圖
	mp 坐標
	buttons 按鈕 0未按下 1左鍵 2右鍵
	clicks 點擊次數
	delta 滾輪值"""
	onClickDefault(view, firstTouch, firstPoint, secondTouch, secondPoint, clicks)
	if view.viewName.find("cycle,") == 0:
		strs = view.viewName.split(",")
		index = int(strs[1])
		cycleInt = int(strs[2])
		myChart = findMyCharts[index]
		myChart.exAttributes["cycle"] = str(cycleInt)
		queryCycle = 0
		if cycleInt > 0 and cycleInt < 1440:
			queryCycle = 1
		elif cycleInt >= 1440:
			queryCycle = 1440
		myCharts = []
		myCharts.append(myChart)
		chart = charts[index]
		code = chart.text.split(" ")[0]
		name = chart.text.split(" ")[1]
		if (chart.viewName == "preChart"):
			chart.datas2 = []
		queryHistoryData(code, name, queryCycle, myCharts)
	elif view.viewName == "preButton":
		if view.text == "預測中...":
			return
		chart = gPaint.findView("preChart")
		thread = threading.Thread(target=predict_in_thread, args=(chart,))
		thread.start()
	elif view.viewType == "menuitem":
		name = view.viewName
		chart = gPaint.findView("preChart")
		if name.startswith("mode"):
			chart.datas2 = []
			chart.preMode = name.split("_")[1]
		elif name.startswith("lookback"):
			chart.lookback = int(name.split("_")[1])
		elif name.startswith("pred_len"):
			chart.datas2 = []
			chart.pred_len = int(name.split("pred_len_")[1])
	elif view.viewName == "temperatureUp":
		chart = gPaint.findView("preChart")
		temperatureDiv = gPaint.findView("temperatureDiv")
		temp = chart.temperature
		if temp < 100:
			chart.temperature += 1.0
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		elif temp < 1:
			chart.temperature += 0.1
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		temperatureDiv.parent.invalidate()
	elif view.viewName == "temperatureDown":
		temperatureDiv = gPaint.findView("temperatureDiv")
		chart = gPaint.findView("preChart")
		temp = chart.temperature
		if temp > 3:
			chart.temperature -= 1.0
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		elif temp > 0.1:
			chart.temperature -= 0.1
			temperatureDiv.text = "Temp:" + str(toFixed(chart.temperature, 1))
		temperatureDiv.parent.invalidate()
	elif view.viewName == "topPUp":
		topPDiv = gPaint.findView("topPDiv")
		chart = gPaint.findView("preChart")
		topP = chart.topP
		if topP < 1:
			chart.topP += 0.1
			topPDiv.text = "topP:" + str(toFixed(chart.topP, 1))
		topPDiv.parent.invalidate()
	elif view.viewName == "topPDown":
		topPDiv = gPaint.findView("topPDiv")
		chart = gPaint.findView("preChart")
		topP = chart.topP
		if topP > 0:
			chart.topP -= 0.1
			topPDiv.text = "topP:" + str(toFixed(chart.topP, 1))
		topPDiv.parent.invalidate()

def createGridCell (grid):
	"""創建單元格"""
	gridCell = FCGridCell()
	if grid.paint.defaultUIStyle == "dark":
		gridCell.backColor = "none"
		gridCell.borderColor = "none"
		gridCell.textColor = "rgb(175,196,228)"
	elif grid.paint.defaultUIStyle == "light":
		gridCell.backColor = "none"
		gridCell.borderColor = "none"
		gridCell.textColor = "rgb(0,0,0)"
	gridCell.font = "Default,13"
	return gridCell

def queryPriceCallBack(data):
	"""板塊數據回調"""
	global priceRowMap
	if data.success:
		global gridStocks
		result = data.data
		strs = result.split("\r\n")
		for i in range(0, len(strs)):
			subStrs = strs[i].split(",")
			if len(subStrs) >= 15:
				row = None
				cell1 = None
				cell2 = None
				cell3 = None
				cell4 = None
				cell5 = None
				cell6 = None
				cell7 = None
				cell8 = None
				cell9 = None
				cell10 = None
				cell11 = None
				cell12 = None
				cell13 = None
				cell14 = None
				cell15 = None
				cell16 = None
				cell17 = None
				cell18 = None
				cell19 = None
				cell20 = None
				cell21 = None
				cell22 = None
				code = subStrs[0]
				if code in priceRowMap:
					row = priceRowMap[code]
					cell1 = row.cells[0]
					cell2 = row.cells[1]
					cell3 = row.cells[2]
					cell4 = row.cells[3]
					cell5 = row.cells[4]
					cell6 = row.cells[5]
					cell7 = row.cells[6]
					cell8 = row.cells[7]
					cell9 = row.cells[8]
					cell10 = row.cells[9]
					cell11 = row.cells[10]
					cell12 = row.cells[11]
					cell13 = row.cells[12]
					cell14 = row.cells[13]
					cell15 = row.cells[14]
					cell16 = row.cells[15]
					cell17 = row.cells[16]
					cell18 = row.cells[17]
					cell19 = row.cells[18]
					cell20 = row.cells[19]
					cell21 = row.cells[20]
					cell22 = row.cells[21]
				else:
					row = FCGridRow()
					priceRowMap[code] = row
					gridStocks.rows.append(row)
					cell1 = createGridCell(gridStocks)
					row.cells.append(cell1)
					cell2 = createGridCell(gridStocks)
					row.cells.append(cell2)
					cell3 = createGridCell(gridStocks)
					row.cells.append(cell3)
					cell4 = createGridCell(gridStocks)
					row.cells.append(cell4)
					cell5 = createGridCell(gridStocks)
					row.cells.append(cell5)
					cell6 = createGridCell(gridStocks)
					row.cells.append(cell6)
					cell7 = createGridCell(gridStocks)
					row.cells.append(cell7)
					cell8 = createGridCell(gridStocks)
					row.cells.append(cell8)
					cell9 = createGridCell(gridStocks)
					row.cells.append(cell9)
					cell10 = createGridCell(gridStocks)
					row.cells.append(cell10)
					cell11 = createGridCell(gridStocks)
					row.cells.append(cell11)
					cell12 = createGridCell(gridStocks)
					row.cells.append(cell12)
					cell13 = createGridCell(gridStocks)
					row.cells.append(cell13)
					cell14 = createGridCell(gridStocks)
					row.cells.append(cell14)
					cell15 = createGridCell(gridStocks)
					row.cells.append(cell15)
					cell16 = createGridCell(gridStocks)
					row.cells.append(cell16)
					cell17 = createGridCell(gridStocks)
					row.cells.append(cell17)
					cell18 = createGridCell(gridStocks)
					row.cells.append(cell18)
					cell19 = createGridCell(gridStocks)
					row.cells.append(cell19)
					cell20 = createGridCell(gridStocks)
					row.cells.append(cell20)
					cell21 = createGridCell(gridStocks)
					row.cells.append(cell21)
					cell22 = createGridCell(gridStocks)
					row.cells.append(cell22)
					cell1.value = len(gridStocks.rows)

					cell2.value = subStrs[0]
					cell2.textColor = "rgb(194,151,18)"
					
					cell3.value = subStrs[1]

				close = float(subStrs[2])
				high = float(subStrs[3])
				low =  float(subStrs[4])
				lastClose = float(subStrs[8])
				cell4.value = toFixed(close, 2)
				cell4.textColor = getPriceColor(close, lastClose)

				diff = 0
				if lastClose > 0:
					diff = 100 * (close - lastClose) / lastClose
				cell5.value = toFixed(diff, 2) + "%"
				cell5.textColor = getPriceColor(diff, 0)

				cell6.value = toFixed(close - lastClose, 2)
				cell6.textColor = getPriceColor(close, lastClose)

				volume = float(subStrs[6])
				amount = float(subStrs[7])
				cell7.value = toFixed(volume / 100 / 10000, 2) + "萬"

				cell8.value = toFixed(amount / 100000000, 2) + "億"

				cell9.value = toFixed(float(subStrs[12]), 2)

				cell10.value = toFixed(float(subStrs[11]), 2)

				diff2 = 0
				if lastClose > 0:
					diff2 = 100 * (high - lastClose) / lastClose - 100 * (low - lastClose) / lastClose
				cell11.value = toFixed(diff2, 2) + "%"

				cell12.value = toFixed(float(subStrs[13]), 2)

				marketValue = float(subStrs[9]) * close
				cell13.value = toFixed(marketValue / 100000000, 2) + "億"

				flowValue = float(subStrs[10]) * close
				cell14.value = toFixed(flowValue / 100000000, 2) + "億"

				cell15.value = ""

				upperLimit = float(subStrs[14])
				lowerLimit = float(subStrs[15])
				cell16.value = toFixed(upperLimit, 2)
				cell16.textColor = getPriceColor(1, 0)

				cell17.value = toFixed(lowerLimit, 2)
				cell17.textColor = getPriceColor(0, 1)

				cell18.value = ""

				cell19.value = ""

				cell20.value = ""

				cell21.value = ""

				cell22.value = ""
		gridStocks.invalidate()

def queryPrice():
	"""查詢報價數據"""
	url = "http://www.jjmfc.com:9968/quote?func=price&codes=all&count=200"
	tag = []
	httpRequest(url, queryPriceCallBack, tag)

def drawChartHScale(chart, paint, clipRect):
	"""繪制橫軸刻度的自定義方法
	chart:圖表
	paint:繪圖對象
	clipRect:裁剪區域"""
	#判斷數據是否為空
	if chart.datas != None and len(chart.datas) > 0 and chart.hScaleHeight > 0:
		if chart.cycle == "trend":
			times = []
			if chart.size.cx < 600:
				times.append(10 * 60 + 30)
				times.append(11 * 60 + 30)
				times.append(14 * 60)
			else:
				times.append(10 * 60)
				times.append(10 * 60 + 30)
				times.append(11 * 60)
				times.append(11 * 60 + 30)
				times.append(13 * 60 + 30)
				times.append(14 * 60)
				times.append(14 * 60 + 30)
			for i in range(chart.firstVisibleIndex, chart.lastVisibleIndex + 1):
				dateNum = chart.datas[i].date
				date = time.localtime(dateNum)
				hour = date.tm_hour
				minute = date.tm_min
				for j in range(0, len(times)):
					if times[j] == hour * 60 + minute:
						x = getChartX(chart, i)
						bBottom = chart.size.cy
						paint.drawLine(chart.scaleColor, 1, 0, x, bBottom - chart.hScaleHeight, x, bBottom - chart.hScaleHeight + 12)
						paint.drawLine(chart.gridColor, 1, 0, x, 0, x, bBottom - chart.hScaleHeight)
						xText = time.strftime("%H:%M", date)
						tSize = paint.textSize(xText, "Default,12")
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", x - tSize.cx / 2, bBottom - chart.hScaleHeight / 2 - tSize.cy / 2)
						break
		elif chart.cycle == "minute":
			lastYear = 0
			lastDate2 = 0
			dLeft = chart.leftVScaleWidth
			i = chart.firstVisibleIndex
			while i <= chart.lastVisibleIndex:
				dateNum = chart.datas[i].date
				date = time.localtime(dateNum)
				year = date.tm_year
				xText = ""
				if year != lastYear:
					xText = time.strftime("%Y/%m/%d", date)
				else:
					xText = time.strftime("%m/%d", date)
				lastDate = time.localtime(lastDate2)
				if int(date.tm_year * 10000 + date.tm_mon * 100 + date.tm_mday) != int(lastDate.tm_year * 10000 + lastDate.tm_mon * 100 + lastDate.tm_mday):
					lastDate2 = dateNum
					lastYear = year
					tSize = paint.textSize(xText, "Default,12")
					x = getChartX(chart, i)
					dx = x + 2
					if dx > dLeft and dx + tSize.cx < chart.size.cx - chart.rightVScaleWidth - 5:
						bBottom = chart.size.cy
						paint.drawLine(chart.scaleColor, 1, 0, x, bBottom - chart.hScaleHeight, x, bBottom - chart.hScaleHeight + 12)
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", dx, bBottom - chart.hScaleHeight / 2 - tSize.cy / 2)
						i = i + int((tSize.cx + chart.hScaleTextDistance) / chart.hScalePixel) + 1
				i = i + 1						
		else:
			drawLeft = chart.leftVScaleWidth #左側起畫點
			i = chart.firstVisibleIndex #開始索引
			lastYear = 0 #緩存年份，用於判斷是否換年
			drawYearsCache = [] #實際繪制到圖形上的年份文字
			lastTextRight = 0 #上個文字的右側
			timeCache = [] #保存日期的緩存
			yearTextLeftCache = [] #繪制年文字的左側位置緩存
			yearTextRightCache = [] #繪制年文字的右側位置緩存
			textPadding = 5 #兩個文字之間的最小間隔
			#逐步遞增索引，先繪制年
			while i <= chart.lastVisibleIndex:
				dateObj = time.localtime(chart.datas[i].date) #將時間戳轉換為time，並緩存到集合中
				timeCache.append(dateObj)
				year = dateObj.tm_year #從結構中獲取年份			
				x = getChartX(chart, i) #獲取索引對應的位置
				#判斷是否換年，以及是否在繪圖區間內
				if year != lastYear and x >= drawLeft and x < chart.size.cx - chart.rightVScaleWidth:
					month = dateObj.tm_mon #獲取月的結構
					xText = str(year) #拼接要繪制的文字
					if month < 10:
						xText = xText + "/0" + str(month) #如果小於10月要補0
					else:
						xText = xText + "/" + str(month) #大於等於10月不用補0
					tSize = paint.textSize(xText, chart.font) #計算要繪制文字的大小
					paint.drawLine(chart.scaleColor, 1, 0, x, chart.size.cy - chart.hScaleHeight, x, chart.size.cy - chart.hScaleHeight + 8) #繪制刻度線
					#判斷是否和上個文字重影
					if x + 2 > lastTextRight + textPadding:
						paint.drawText(xText, chart.hScaleTextColor, "Default,12", x + 2, chart.size.cy - chart.hScaleHeight + 8  - tSize.cy / 2 + 7) #繪制文字
						yearTextLeftCache.append(x + 2) #將年文字的左側位置緩存
						yearTextRightCache.append(x + 2 + tSize.cx) #將年文字的右側位置緩存
						drawYearsCache.append(year) #緩存要繪制的年
						lastTextRight = x + 2 + tSize.cx #緩存上個文字的右側位置
					lastYear = year #記錄上次繪制的年份
				i = i + 1	#索引累加	
			#繪制月份
			for m in range(0, len(drawYearsCache)):
				cacheYear = drawYearsCache[m] #從緩存中獲取年份
				lastMonth = 0 #緩存月份，用於判斷是否換月
				i = chart.firstVisibleIndex #重置開始索引
				lastTextRight = 0 #重置上個文字的右側
				#逐步遞增索引
				while i <= chart.lastVisibleIndex:
					dateObj = timeCache[i - chart.firstVisibleIndex] #從緩存中獲取time
					year = dateObj.tm_year #從結構中獲取年份
					#判斷是否同一年	
					if cacheYear == year:
						month = dateObj.tm_mon #從結構中獲取月份
						x = getChartX(chart, i)
						#判斷是否換月，以及是否在繪圖區間內
						if lastMonth != month and x >= drawLeft and x < chart.size.cx - chart.rightVScaleWidth:			
							xText = str(month) #獲取繪制的月份文字
							tSize = paint.textSize(xText, chart.font) #計算要繪制文字的大小
							#判斷是否和上個文字重影
							if x + 2 > lastTextRight + textPadding:
								#判斷是否和年的文字重影
								if (x + 2 > yearTextRightCache[m] + textPadding) and ((m == len(drawYearsCache) - 1) or (m < len(drawYearsCache) - 1 and x + 2 + tSize.cx < yearTextLeftCache[m + 1] - textPadding)):
									paint.drawLine(chart.scaleColor, 1, 0, x, chart.size.cy - chart.hScaleHeight, x, chart.size.cy - chart.hScaleHeight + 6) #繪制刻度
									paint.drawText(xText, chart.hScaleTextColor, "Default,12", x + 2, chart.size.cy - chart.hScaleHeight + 8  - tSize.cy / 2 + 7) #繪制文字
									lastTextRight = x + 2 + tSize.cx #緩存上個文字的右側位置
							lastMonth = month #記錄上次繪制的月份
					elif cacheYear < year:
						break #超過區間，退出循環
					i = i + 1	#索引累加

def drawChartTip(chart, paint, clipRect):
	"""繪制圖表提示
	chart:圖表
	paint:繪圖對象
	clipRect:裁剪區域"""
	if paint.touchMoveView == chart and chart.cycle != "trend":
		crossLineIndex = chart.crossStopIndex
		if crossLineIndex != -1 and crossLineIndex >= chart.firstVisibleIndex and crossLineIndex <= chart.lastVisibleIndex:
			mp = chart.touchPosition
			cmp = FCPoint(mp.x + 5, mp.y)
			width = 125
			height = 125
			tipRect = FCRect(cmp.x, cmp.y, cmp.x + width, cmp.y + height)
			if tipRect.left < 0:
				tipRect.left = 0
				tipRect.right = width
			if tipRect.right > chart.size.cx:
				tipRect.left = chart.size.cx - width
				tipRect.right = tipRect.left + width
			if tipRect.bottom > chart.size.cy:
				tipRect.top = chart.size.cy - height
				tipRect.bottom = tipRect.top + height
			sData = chart.datas[crossLineIndex]
			high = sData.high
			low = sData.low
			highY = getChartY(chart, 0, high)
			lowY = getChartY(chart, 0, low)
			if cmp.y >= highY and cmp.y <= lowY:
				if paint.defaultUIStyle == "dark":
					paint.fillRect("rgb(50,50,50)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				elif paint.defaultUIStyle == "light":
					paint.fillRect("rgb(220,220,220)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				close = sData.close
				openPrice = sData.open
				volume = sData.volume
				lastClose = sData.open
				if crossLineIndex > 0:
					lastClose = chart.datas[crossLineIndex - 1].close
				xText = ""
				if chart.cycle == "day":
					timeArray = time.localtime(sData.date)
					xText = time.strftime("%Y-%m-%d", timeArray)
				elif chart.cycle == "minute":
					timeArray = time.localtime(sData.date)
					xText = time.strftime("%Y-%m-%d %H:%M", timeArray)
				xFont = "Default,13"
				paint.drawText(xText, chart.textColor, xFont, tipRect.left + 5, tipRect.top + 5)
				paint.drawText("高:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 25)
				paint.drawText(toFixed(high, chart.candleDigit), getPriceColor(high, lastClose), xFont, tipRect.left + 25, tipRect.top + 25)
				paint.drawText("開:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 45)
				paint.drawText(toFixed(openPrice, chart.candleDigit), getPriceColor(openPrice, lastClose), xFont, tipRect.left + 25, tipRect.top + 45)
				paint.drawText("低:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 65)
				paint.drawText(toFixed(low, chart.candleDigit), getPriceColor(low, lastClose), xFont, tipRect.left + 25, tipRect.top + 65)
				paint.drawText("收:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 85)
				paint.drawText(toFixed(close, chart.candleDigit), getPriceColor(close, lastClose), xFont, tipRect.left + 25, tipRect.top + 85)
				paint.drawText("量:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 105)
				paint.drawText(toFixed(volume, 0), "rgb(80,255,255)", xFont, tipRect.left + 25, tipRect.top + 105)

def WndProcPopup(hwnd,msg,wParam,lParam):
	"""消息循環"""
	global popupPaint
	if (hwnd in popupPaint) == True:
		return WndProcDefault(popupPaint[hwnd],hwnd,msg,wParam,lParam)
	else:
		return DefWindowProc(hwnd,msg,wParam,lParam)

popupPaint = dict()
# 子窗體
def popupWindow(text):
	global popupPaint
	pPaint = FCPaint() #創建繪圖對象
	wSize = FCSize(500, 200)
	setWindowSize(wSize)
	setCenterScreen(True)
	#初始化窗體
	createWindow(pPaint, "提示", WndProcPopup)
	popupPaint[pPaint.hWnd] = pPaint
	# 轉義文本，避免XML注入/渲染失敗
	safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
	xml = f"""<?xml version="1.0" encoding="UTF-8"?>
		<html xmlns="facecat">
		<body>
			<div dock="fill" backcolor="rgb(170,178,189)">
				<label name="labelText" size="350,33" text="{safe_text}" font="Default,30" location="100,20"/>
				<label name="labelText2" size="350,33" text="請切換周期或修改訓練參數。" font="Default,30" location="100,55"/>
			</div>
		</body>
		</html>"""
	renderFaceCat(pPaint, xml)
	showWindow(pPaint)
	# 模態阻塞：直到窗口被關閉
	msg = cwintypes.MSG()
	pmsg = ct.byref(msg)
	while user32.IsWindow(pPaint.hWnd):
		res = user32.GetMessageW(pmsg, None, 0, 0)
		if res <= 0:
			break
		user32.TranslateMessage(pmsg)
		user32.DispatchMessageW(pmsg)

MAIN_XML = """<?xml version="1.0" encoding="utf-8" ?>
<html xmlns="facecat">
<body>
	<div bordercolor="none" name="divInner" dock="fill">
		<div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none" name="tabFunc">
			<div type="tabpage" text="主畫面" name="divMain" backcolor="none">
				<div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill" size="650,510" candragsplitter="true" splitterposition="650,1" name="divCodingRight2">
					<table name="gridStocks" headerheight="30" dock="fill" gridlinecolor="none" bordercolor="none" showvscrollbar="true" showhscrollbar="true" allowpreviewsevent="true" allowdragscroll="true">
						<tr>
							<th name="colP0" text="序" width="40" allowdrag="true" allowresize="true" coltype="no"/>
							<th name="colP1" text="代碼" width="70" allowdrag="true" allowresize="true"/>
							<th name="colP2" text="名稱" width="90" allowdrag="true" allowresize="true"/>
							<th name="colP3" text="現價" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP4" text="漲幅" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP5" text="漲跌" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP6" text="開盤" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP7" text="最高" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP8" text="最低" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP9" text="成交量" width="90" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP10" text="昨收" width="70" allowdrag="true" allowresize="true" cellalign="right"/>
							<th name="colP11" text="分類" width="90" allowdrag="true" allowresize="true" cellalign="center"/>
						</tr>
					</table>
					<div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none" name="tabFunc">
						<div type="tabpage" text="AI 預測" name="preTab" backcolor="none">
							<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="25,1">
								<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="300,1" splitmode="percentsize">
									<div type="layout" backcolor="none" allowResize="false" bordercolor="none">
										<input type="button" name="preButton" text="預測" textcolor="rgb(255,255,255)" />
										<select name="modeSwich" selectedindex="0">
											<option text="預測模式" name="mode_predict" value="predict"/>
											<option text="回測模式" name="mode_backtest" value="backtest"/>
										</select>
									</div>
									<div type="layout" backcolor="none" bordercolor="none">
										<select name="lookback" selectedindex="0" size="90,25">
											<option text="樣本 50" name="lookback_50" value="50"/>
											<option text="樣本 10" name="lookback_10" value="10"/>
											<option text="樣本 20" name="lookback_20" value="20"/>
											<option text="樣本 100" name="lookback_100" value="100"/>
											<option text="樣本 200" name="lookback_200" value="200"/>
											<option text="樣本 300" name="lookback_300" value="300"/>
											<option text="樣本 400" name="lookback_400" value="400"/>
										</select>
										<select name="pred_len" size="90,25">
											<option text="預測 5" name="pred_len_5" value="5"/>
											<option text="預測 1" name="pred_len_1" value="1"/>
											<option text="預測 10" name="pred_len_10" value="10"/>
											<option text="預測 20" name="pred_len_20" value="20"/>
											<option text="預測 50" name="pred_len_50" value="50"/>
											<option text="預測 120" name="pred_len_120" value="120"/>
										</select>
										<div size="90,25" name="temperatureDiv" text="Temp:1.0">
											<input type="button" location="72,4" size="10,12" name="temperatureUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
											<input type="button" location="72,8" size="10,12" name="temperatureDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
										</div>
										<div size="90,25" name="topPDiv" text="topP:0.9">
											<input type="button" location="72,4" size="10,12" name="topPUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
											<input type="button" location="72,8" size="10,12" name="topPDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
										</div>
										<div name="progress" bordercolor="none" text="0"/>
									</div>
								</div>
								<div type="custom" cid="mychart" name="preDiv" cycle="1440" size="200,200" nativerefresh="true" candledivpercent="0.7" voldivpercent="0.3"/>
							</div>
						</div>
						<div type="tabpage" text="走勢圖" name="quoteTab" backcolor="none">
							<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="380,1">
								<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="200,1">
									<div type="custom" cid="latestdiv" name="divLatest"/>
									<div type="custom" cid="mychart" name="mainChart1" cycle="0" candledivpercent="0.7" voldivpercent="0.3" backcolor="none" bordercolor="none"/>
								</div>
								<div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
									<div type="custom" cid="mychart" name="mainChart2" cycle="5" candledivpercent="0.7" voldivpercent="0.3" backcolor="none" bordercolor="none"/>
									<div type="custom" cid="mychart" name="mainChart3" cycle="1440" candledivpercent="0.7" voldivpercent="0.3" backcolor="none" bordercolor="none"/>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			<div type="tabpage" text="多 K 線" name="divMulti" backcolor="none">
				<div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="133,1">
					<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
						<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
							<div type="custom" cid="mychart" name="chart1" cycle="1" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
							<div type="custom" cid="mychart" name="chart2" cycle="5" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
						</div>
						<div type="custom" cid="mychart" name="chart3" cycle="10" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
					</div>
					<div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
						<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
							<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
								<div type="custom" cid="mychart" name="chart4" cycle="15" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
								<div type="custom" cid="mychart" name="chart5" cycle="20" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
							</div>
							<div type="custom" cid="mychart" name="chart6" cycle="30" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
						</div>
						<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
							<div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
								<div type="custom" cid="mychart" name="chart7" cycle="1440" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
								<div type="custom" cid="mychart" name="chart8" cycle="10080" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
							</div>
							<div type="custom" cid="mychart" name="chart9" cycle="43200" nativerefresh="true" candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</body>
</html>"""


def drawProgressDiv(view, paint, clipRect):
	if view.text == "-1":
		paint.drawText("預測中...", "rgb(255,255,255)", "Default,14", 3, 7)
	else:
		cx = float(view.text) * view.size.cx
		if cx > 0 and cx < view.size.cx:
			paint.fillRect(view.textColor, 0, 0, cx, view.size.cy)


def drawLatestDiv(view, paint, clipRect):
	global latestQuote
	avgHeight = 20
	drawFont = "Default,14"
	textColor = "rgb(175,196,228)"
	if view.paint.defaultUIStyle == "light":
		textColor = "rgb(0,0,0)"
	dTop = 30
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	last_close = safe_float(latestQuote.get("last_close"))
	sell_book = latestQuote.get("sell_book", [])
	buy_book = latestQuote.get("buy_book", [])
	book = sell_book + buy_book
	labels = ["賣5", "賣4", "賣3", "賣2", "賣1", "買1", "買2", "買3", "買4", "買5"]
	if latestQuote:
		paint.drawText(latestQuote.get("code", ""), textColor, drawFont, 5, 7)
		paint.drawText(latestQuote.get("name", ""), "rgb(194,151,18)", drawFont, 80, 7)
	for i in range(0, 10):
		tSize = paint.textSize(labels[i], drawFont)
		paint.drawText(labels[i], textColor, drawFont, 5, dTop + avgHeight / 2 - tSize.cy / 2)
		if i < len(book):
			price, volume = book[i]
			price_color = "rgb(255,82,82)"
			if price < last_close:
				price_color = "rgb(46,255,50)"
				if paint.defaultUIStyle == "light":
					price_color = "rgb(0,200,0)"
			paint.drawText(toFixed(price, 2), price_color, drawFont, 50, dTop + avgHeight / 2 - tSize.cy / 2)
			vol_text = toFixed(volume, 0)
			vol_size = paint.textSize(vol_text, drawFont)
			paint.drawText(vol_text, textColor, drawFont, view.size.cx - vol_size.cx - 10, dTop + avgHeight / 2 - vol_size.cy / 2)
		dTop += avgHeight
	paint.drawLine(view.borderColor, 1, 0, 0, dTop, view.size.cx, dTop)
	paint.drawText("現價", textColor, drawFont, 5, dTop + 10)
	paint.drawText("漲幅", textColor, drawFont, 5, dTop + 35)
	paint.drawText("成交額", textColor, drawFont, 5, dTop + 60)
	paint.drawText("成交量", textColor, drawFont, 5, dTop + 85)
	paint.drawText("開盤", textColor, drawFont, 110, dTop + 10)
	paint.drawText("振幅", textColor, drawFont, 110, dTop + 35)
	paint.drawText("最高", textColor, drawFont, 110, dTop + 60)
	paint.drawText("最低", textColor, drawFont, 110, dTop + 85)
	if latestQuote:
		close = safe_float(latestQuote.get("close"))
		high = safe_float(latestQuote.get("high"))
		low = safe_float(latestQuote.get("low"))
		open_price = safe_float(latestQuote.get("open"))
		volume = safe_float(latestQuote.get("volume"))
		amount = safe_float(latestQuote.get("amount"))
		diff = 0
		diff2 = 0
		if last_close > 0:
			diff = 100 * (close - last_close) / last_close
			diff2 = 100 * (high - last_close) / last_close - 100 * (low - last_close) / last_close
		paint.drawText(toFixed(close, 2), getPriceColor(close, last_close), drawFont, 45, dTop + 10)
		paint.drawText(toFixed(diff, 2) + "%", getPriceColor(close, last_close), drawFont, 45, dTop + 35)
		paint.drawText(toFixed(amount / 100000000, 2) + "億", textColor, drawFont, 45, dTop + 60)
		paint.drawText(toFixed(volume / 1000, 0) + "張", textColor, drawFont, 45, dTop + 85)
		paint.drawText(toFixed(open_price, 2), getPriceColor(open_price, last_close), drawFont, 145, dTop + 10)
		paint.drawText(toFixed(diff2, 2) + "%", getPriceColor(close, last_close), drawFont, 145, dTop + 35)
		paint.drawText(toFixed(high, 2), getPriceColor(high, last_close), drawFont, 145, dTop + 60)
		paint.drawText(toFixed(low, 2), getPriceColor(low, last_close), drawFont, 145, dTop + 85)


def setChartTitle(chart, code, name, intCycle):
	chart.text = code + " " + name
	if intCycle == 0:
		chart.text += " 分時"
	elif intCycle < 1440:
		chart.text += " " + str(intCycle) + "分鐘"
	elif intCycle == 1440:
		chart.text += " 日線"
	elif intCycle == 10080:
		chart.text += " 週線"
	elif intCycle == 43200:
		chart.text += " 月線"
	elif intCycle == 129600:
		chart.text += " 季線"
	elif intCycle == 259200:
		chart.text += " 半年線"
	elif intCycle == 518400:
		chart.text += " 年線"


def historyDataCallBack(data):
	if data.success:
		code, name, cycle, myCharts = data.tag
		base_datas = parse_yahoo_history(data.data)
		if not base_datas:
			return
		for c in range(0, len(myCharts)):
			myChart = myCharts[c]
			myCycle = int(myChart.exAttributes["cycle"])
			chart = myChart.views[0].secondView.views[0]
			copyDatas = [copySecurityData(item) for item in base_datas]
			if cycle == 0 and myCycle == 0:
				chart.autoFillHScale = True
				chart.cycle = "trend"
			elif cycle == 1 and (myCycle > 0 and myCycle < 1440):
				chart.cycle = "minute"
				if myCycle > 1:
					newDatas = []
					multiMinuteSecurityDatas(newDatas, copyDatas, myCycle)
					copyDatas = newDatas
			elif cycle == 1440 and myCycle >= 1440:
				chart.cycle = "day"
				if myCycle == 10080:
					newDatas = []
					getHistoryWeekDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 43200:
					newDatas = []
					getHistoryMonthDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 129600:
					newDatas = []
					getHistorySeasonDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 259200:
					newDatas = []
					getHistoryHalfYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
				elif myCycle == 518400:
					newDatas = []
					getHistoryYearDatas(newDatas, copyDatas)
					copyDatas = newDatas
			else:
				continue
			clientTicks[c] = ClientTickDataCache()
			setChartTitle(chart, code, name, myCycle)
			chart.firstOpen = copyDatas[0].open if copyDatas else 0
			chart.lastVisibleKey = 0
			chart.firstVisibleIndex = -1
			chart.lastVisibleIndex = -1
			chart.datas = copyDatas
			maxVisibleRecord = getChartMaxVisibleCount(chart, chart.hScalePixel, getChartWorkAreaWidth(chart))
			chart.lastVisibleIndex = len(chart.datas) - 1
			if maxVisibleRecord > len(chart.datas):
				chart.firstVisibleIndex = 0
			else:
				chart.firstVisibleIndex = chart.lastVisibleIndex - maxVisibleRecord + 1
			resetChartVisibleRecord(chart)
			checkChartLastVisibleIndex(chart)
			calcChartIndicator(chart)
			chart.invalidate()


def queryHistoryData(code, name, cycle, myCharts):
	if cycle == 0:
		url = build_yahoo_chart_url(code, "1m", "8d")
	elif cycle < 1440:
		url = build_yahoo_chart_url(code, "1m", "8d")
	else:
		url = build_yahoo_chart_url(code, "1d", "10y")
	httpRequest(url, historyDataCallBack, [code, name, cycle, myCharts])


def newDataCallBack(data):
	if data.success:
		global latestQuote
		payload = json.loads(data.data)
		msg_array = payload.get("msgArray", [])
		if not msg_array:
			return
		latestQuote = build_latest_quote(msg_array[0])
		code = latestQuote.get("code", "")
		close = safe_float(latestQuote.get("close"))
		volume = safe_float(latestQuote.get("volume")) * 1000
		amount = safe_float(latestQuote.get("amount"))
		date_value = safe_float(latestQuote.get("time"))
		date = int(date_value / 1000) if date_value > 0 else int(time.time())
		for c in range(0, len(findMyCharts)):
			myChart = findMyCharts[c]
			chart = charts[c]
			if chart.viewName == "preChart":
				continue
			if code == chart.text.split(" ")[0] and len(chart.datas) > 0:
				myCycle = int(myChart.exAttributes["cycle"])
				latestData = SecurityData()
				latestData.close = close
				latestData.open = close
				latestData.high = close
				latestData.low = close
				latestData.volume = volume
				latestData.amount = amount
				latestData.date = date
				cTick = clientTicks.get(c, ClientTickDataCache())
				if len(cTick.code) == 0:
					cTick.code = currentCode
					cTick.lastAmount = latestData.amount
					cTick.lastVolume = latestData.volume
					cTick.lastDate = latestData.date
				oldDataSize = len(chart.datas)
				mergeLatestData(code, chart.datas, latestData, cTick, myCycle)
				clientTicks[c] = cTick
				if myCycle == 0:
					for d in range(0, len(chart.datas)):
						if chart.datas[d].volume > 0:
							chart.lastValidIndex = d
				if chart.lastRecordIsVisible and len(chart.datas) > oldDataSize:
					chart.firstVisibleIndex += 1
					chart.lastVisibleIndex += 1
				checkChartLastVisibleIndex(chart)
				calcChartIndicator(chart)
		gPaint.update()


def queryNewData():
	httpRequest(build_mis_url([currentCode]), newDataCallBack, [])


def queryPriceCallBack(data):
	global priceRowMap
	if data.success:
		payload = json.loads(data.data)
		msg_array = payload.get("msgArray", [])
		for stock_info in msg_array:
			quote = build_latest_quote(stock_info)
			code = quote["code"]
			row = priceRowMap.get(code)
			if row is None:
				row = FCGridRow()
				priceRowMap[code] = row
				gridStocks.rows.append(row)
				for _ in range(12):
					row.cells.append(createGridCell(gridStocks))
				row.cells[0].value = len(gridStocks.rows)
			row.cells[1].value = code
			row.cells[1].textColor = "rgb(194,151,18)"
			row.cells[2].value = quote["name"]
			close = safe_float(quote["close"])
			last_close = safe_float(quote["last_close"])
			open_price = safe_float(quote["open"])
			high = safe_float(quote["high"])
			low = safe_float(quote["low"])
			volume = safe_float(quote["volume"])
			diff = close - last_close
			diff_pct = 100 * diff / last_close if last_close > 0 else 0
			row.cells[3].value = toFixed(close, 2)
			row.cells[3].textColor = getPriceColor(close, last_close)
			row.cells[4].value = toFixed(diff_pct, 2) + "%"
			row.cells[4].textColor = getPriceColor(diff, 0)
			row.cells[5].value = toFixed(diff, 2)
			row.cells[5].textColor = getPriceColor(diff, 0)
			row.cells[6].value = toFixed(open_price, 2)
			row.cells[7].value = toFixed(high, 2)
			row.cells[8].value = toFixed(low, 2)
			row.cells[9].value = toFixed(volume / 1000, 0) + "張"
			row.cells[10].value = toFixed(last_close, 2)
			row.cells[11].value = quote["category"]
		gridStocks.invalidate()


def queryPrice():
	httpRequest(build_mis_url([item["code"] for item in TAIWAN_WATCHLIST]), queryPriceCallBack, [])


def drawChartTip(chart, paint, clipRect):
	if paint.touchMoveView == chart and chart.cycle != "trend":
		crossLineIndex = chart.crossStopIndex
		if crossLineIndex != -1 and chart.firstVisibleIndex <= crossLineIndex <= chart.lastVisibleIndex:
			mp = chart.touchPosition
			cmp = FCPoint(mp.x + 5, mp.y)
			width = 125
			height = 125
			tipRect = FCRect(cmp.x, cmp.y, cmp.x + width, cmp.y + height)
			if tipRect.left < 0:
				tipRect.left = 0
				tipRect.right = width
			if tipRect.right > chart.size.cx:
				tipRect.left = chart.size.cx - width
				tipRect.right = tipRect.left + width
			if tipRect.bottom > chart.size.cy:
				tipRect.top = chart.size.cy - height
				tipRect.bottom = tipRect.top + height
			sData = chart.datas[crossLineIndex]
			high = sData.high
			low = sData.low
			highY = getChartY(chart, 0, high)
			lowY = getChartY(chart, 0, low)
			if cmp.y >= highY and cmp.y <= lowY:
				if paint.defaultUIStyle == "dark":
					paint.fillRect("rgb(50,50,50)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				else:
					paint.fillRect("rgb(220,220,220)", tipRect.left, tipRect.top, tipRect.right, tipRect.bottom)
				close = sData.close
				openPrice = sData.open
				volume = sData.volume
				lastClose = chart.datas[crossLineIndex - 1].close if crossLineIndex > 0 else sData.open
				xText = time.strftime("%Y-%m-%d", time.localtime(sData.date))
				if chart.cycle == "minute":
					xText = time.strftime("%Y-%m-%d %H:%M", time.localtime(sData.date))
				xFont = "Default,13"
				paint.drawText(xText, chart.textColor, xFont, tipRect.left + 5, tipRect.top + 5)
				paint.drawText("高:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 25)
				paint.drawText(toFixed(high, chart.candleDigit), getPriceColor(high, lastClose), xFont, tipRect.left + 25, tipRect.top + 25)
				paint.drawText("開:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 45)
				paint.drawText(toFixed(openPrice, chart.candleDigit), getPriceColor(openPrice, lastClose), xFont, tipRect.left + 25, tipRect.top + 45)
				paint.drawText("低:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 65)
				paint.drawText(toFixed(low, chart.candleDigit), getPriceColor(low, lastClose), xFont, tipRect.left + 25, tipRect.top + 65)
				paint.drawText("收:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 85)
				paint.drawText(toFixed(close, chart.candleDigit), getPriceColor(close, lastClose), xFont, tipRect.left + 25, tipRect.top + 85)
				paint.drawText("量:", chart.textColor, xFont, tipRect.left + 5, tipRect.top + 105)
				paint.drawText(toFixed(volume, 0), "rgb(80,255,255)", xFont, tipRect.left + 25, tipRect.top + 105)


def popupWindow(text):
	global popupPaint
	pPaint = FCPaint()
	wSize = FCSize(500, 200)
	setWindowSize(wSize)
	setCenterScreen(True)
	createWindow(pPaint, "提示", WndProcPopup)
	popupPaint[pPaint.hWnd] = pPaint
	safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
	xml = f"""<?xml version="1.0" encoding="UTF-8"?>
		<html xmlns="facecat">
		<body>
			<div dock="fill" backcolor="rgb(170,178,189)">
				<label name="labelText" size="420,33" text="{safe_text}" font="Default,24" location="40,30"/>
				<label name="labelText2" size="420,33" text="請切換週期或調整預測參數。" font="Default,24" location="40,70"/>
			</div>
		</body>
		</html>"""
	renderFaceCat(pPaint, xml)
	showWindow(pPaint)
	msg = cwintypes.MSG()
	pmsg = ct.byref(msg)
	while user32.IsWindow(pPaint.hWnd):
		res = user32.GetMessageW(pmsg, None, 0, 0)
		if res <= 0:
			break
		user32.TranslateMessage(pmsg)
		user32.DispatchMessageW(pmsg)

def normalize_prediction_params(chart):
	"""Fit lookback/pred_len to the loaded sample size before predicting."""
	available = len(chart.datas)
	mode = chart.preMode
	lookback = int(chart.lookback)
	pred_len = int(chart.pred_len)
	adjustments = []
	min_lookback = 10

	if available < min_lookback:
		return None, None, f"K 線數量不足，目前只有 {available} 筆，至少需要 {min_lookback} 筆。"

	max_lookback = available if mode == "predict" else max(min_lookback, available - 1)
	if lookback > max_lookback:
		lookback = max_lookback
		adjustments.append(f"樣本改為 {lookback}")

	if mode == "backtest":
		max_pred_len = max(1, available - lookback)
		if pred_len > max_pred_len:
			pred_len = max_pred_len
			adjustments.append(f"預測改為 {pred_len}")

	if lookback < min_lookback:
		return None, None, f"K 線數量不足，目前只有 {available} 筆，至少需要 {min_lookback} 筆。"

	message = ""
	if adjustments:
		message = "資料筆數不足，已自動調整參數：" + "，".join(adjustments) + f"。目前共 {available} 筆資料。"
	return lookback, pred_len, message


def ensure_predictor_loaded():
	"""Load predictor lazily and auto-download missing Hugging Face weights."""
	global tokenizer
	global model
	global predictor
	if predictor is not None:
		return predictor
	try:
		tokenizer_dir = ensure_local_pretrained_dir(LOCAL_TOKENIZER_DIR, HF_TOKENIZER_REPO)
		model_dir = ensure_local_pretrained_dir(LOCAL_MODEL_DIR, HF_MODEL_REPO)
		tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir)
		model = Kronos.from_pretrained(model_dir)
		predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
		print(f"[startup] model ready on {device}")
		return predictor
	except Exception as ex:
		popupWindow("AI 模型載入失敗。\n本地模型檔不完整，程式已嘗試自動下載；若仍失敗，請確認網路連線或模型資料夾權限。\n\n詳細錯誤: " + str(ex))
		return None


def predict(chart):
	local_predictor = ensure_predictor_loaded()
	if local_predictor is None:
		return
	preButton = gPaint.findView("preButton")
	preButton.enabled = False
	progressDiv = gPaint.findView("progress")
	progressDiv.text = "-1"
	progressDiv.invalidate()
	preButton.invalidate()
	lookback, pred_len, adjust_message = normalize_prediction_params(chart)
	mode = chart.preMode
	if lookback is None:
		popupWindow(adjust_message)
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	if adjust_message:
		popupWindow(adjust_message)
	chart.lookback = lookback
	chart.pred_len = pred_len
	if len(chart.datas) < lookback and mode == "predict":
		popupWindow(f"K 線數量不足，目前只有 {len(chart.datas)} 筆，至少需要 {lookback} 筆。")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	elif len(chart.datas) < lookback + pred_len and mode == "backtest":
		popupWindow(f"K 線數量不足，目前只有 {len(chart.datas)} 筆，至少需要 {lookback + pred_len} 筆。")
		progressDiv.text = "0"
		progressDiv.invalidate()
		preButton.enabled = True
		preButton.invalidate()
		return
	if mode == "predict":
		klines = chart.lastVisibleIndex - chart.firstVisibleIndex
		if chart.pred_len > 50 and klines > 50:
			chart.firstVisibleIndex += 50
		elif chart.pred_len <= 50 and klines > chart.pred_len:
			chart.firstVisibleIndex += chart.pred_len
		chart.invalidate()
		df = transToPanda(chart.datas)
		df["timestamps"] = pd.to_datetime(df["timestamps"])
		df = df.sort_values("timestamps").reset_index(drop=True)
		x_df = df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
		x_timestamp = df.tail(lookback)["timestamps"].reset_index(drop=True)
		last_timestamp = x_timestamp.iloc[-1]
		y_timestamp = pd.Series(pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=pred_len))
		pred_df = local_predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		chart.datas2 = transToChartData(pred_df)
	else:
		df = transToPanda(chart.datas)
		df["timestamps"] = pd.to_datetime(df["timestamps"])
		df = df.sort_values("timestamps").reset_index(drop=True)
		test_df = df.tail(lookback + pred_len).reset_index(drop=True)
		x_df = test_df.loc[:lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
		x_timestamp = test_df.loc[:lookback - 1, "timestamps"]
		y_timestamp = test_df.loc[lookback:lookback + pred_len - 1, "timestamps"]
		pred_df = local_predictor.predict(
			df=x_df,
			x_timestamp=x_timestamp,
			y_timestamp=y_timestamp,
			pred_len=pred_len,
			T=chart.temperature,
			top_p=chart.topP,
			sample_count=1,
			verbose=True,
			progress_callback=progress_callback
		)
		chart.datas2 = transToChartData(pred_df)
	preButton.enabled = True
	preButton.invalidate()


def predict_in_thread(chart):
	predict(chart)
	chart.invalidate()
def WndProc(hwnd,msg,wParam,lParam):
	"""消息循環"""
	if msg == 0x0401:
		gPaint.dealData()
	return WndProcDefault(gPaint,hwnd,msg,wParam,lParam)

gPaint = FCPaint() #創建繪圖對象
gPaint.defaultUIStyle = "dark"
gPaint.onPaint = onPaint
gPaint.onClickGridCell = onClickGridCell
gPaint.onClick = onClick
gPaint.onPaintChartHScale = drawChartHScale

gPaint.highQuanlity()
gPaint.scaleFactorX = 1.33
gPaint.scaleFactorY = 1.33

#初始化窗體
createMainWindow(gPaint, "Facecat-Kronos", WndProc)
xml = """<?xml version="1.0" encoding="utf-8" ?>
	    <html xmlns="facecat">
	    <body>
	        <div bordercolor="none" name="divInner" dock="fill">
	            <div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none"
	            name="tabFunc">
	                <div type="tabpage" text="主界面" name="divMain" backcolor="none">
	                <div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill"
	                        size="650,510" candragsplitter="true" splitterposition="650,1" name="divCodingRight2">
	                        <table name="gridStocks" headerheight="30" dock="fill"
	                            gridlinecolor="none" bordercolor="none" showvscrollbar="true" showhscrollbar="true"
	                            allowpreviewsevent="true" allowdragscroll="true">
	                            <tr>
	                                <th name="colP0" text="序" width="40" allowdrag="true" allowresize="true" coltype="no"/>
	                                <th name="colP1" text="代碼" width="70" allowdrag="true" allowresize="true"/>
	                                <th name="colP2" text="名稱" width="70" allowdrag="true" allowresize="true" />
	                                <th name="colP3" text="現價" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP4" text="漲幅" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP5" text="漲跌" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP9" text="總量" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP10" text="總額" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP11" text="量比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP12" text="PE動" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP13" text="振幅" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP14" text="換手率" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP15" text="總市值" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP16" text="流值" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP17" text="行業分類板塊" width="80" allowdrag="true" allowresize="true" cellalign="center"/>
	                                <th name="colP18" text="漲停價" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP19" text="跌停價" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP20" text="金比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP21" text="漲跌比" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP22" text="漲速" width="60" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP23" text="淨資產收益率" width="100" allowdrag="true" allowresize="true" cellalign="right"/>
	                                <th name="colP24" text="自設指標" width="80" allowdrag="true" allowresize="true" cellalign="right"/>
	                            </tr>
	                        </table>
							<div type="tab" dock="fill" selectedindex="0" backcolor="none" bordercolor="none"
	           					 name="tabFunc">
								<div type="tabpage" text="預測" name="preTab" backcolor="none">
									<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="25,1" >
										<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill" size="400,400" candragsplitter="true" splitterposition="300,1" splitmode="percentsize">
											<div type="layout" backcolor="none" allowResize="false" bordercolor="none">
												<input type="button" name="preButton" text="預測" textcolor="rgb(255,255,255)" /> 
												<select name="modeSwich" selectedindex="0">
													<option text="預測模式" name="mode_predict" value="predict"/>
													<option text="回測模式" name="mode_backtest" value="backtest"/>
												</select>
											</div>
											<div type="layout" backcolor="none" bordercolor="none">
												<select name="lookback" selectedindex="0" size="80,25">
													<option text="訓練 50" name="lookback_50" value="50"/>
													<option text="訓練 10" name="lookback_10" value="10"/>
													<option text="訓練 20" name="lookback_20" value="20"/>
													<option text="訓練 100" name="lookback_100" value="100"/>
													<option text="訓練 200" name="lookback_200" value="200"/>
													<option text="訓練 300" name="lookback_300" value="300"/>
													<option text="訓練 400" name="lookback_400" value="400"/>
												</select>
												<select name="pred_len" size="80,25">
													<option text="預測 5" name="pred_len_5" value="5"/>
													<option text="預測 1" name="pred_len_1" value="1"/>
													<option text="預測 10" name="pred_len_10" value="10"/>
													<option text="預測 20" name="pred_len_20" value="20"/>
													<option text="預測 50" name="pred_len_50" value="50"/>
													<option text="預測 120" name="pred_len_120" value="120"/>
												</select>
												<div size="80,25" name="temperatureDiv" text="Temp:1.0">
													<input type="button" location="65,4" size="10,12" name="temperatureUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
													<input type="button" location="65,8" size="10,12" name="temperatureDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
												</div>
												<div size="80,25" name="topPDiv" text="topP:0.9">
													<input type="button" location="65,4" size="10,12" name="topPUp" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
													<input type="button" location="65,8" size="10,12" name="topPDown" hoveredColor="rgb(180,180,180)" pushedColor="rgb(100,100,100)"/>
												</div>
												<div name="progress" bordercolor="none" text="0" />
											</div>
											
										</div>
										<div type="custom" cid="mychart" name="preDiv" cycle="1440" size="200,200" nativerefresh="true" 
										     candledivpercent="0.7"  voldivpercent="0.3"  />
									</div>
								</div>
								<div type="tabpage" text="主界面" name="divMain" backcolor="none">
									<div type="splitlayout" layoutstyle="toptobottom" bordercolor="none" dock="fill"
										size="400,400" candragsplitter="true" splitterposition="380,1">
										<div type="splitlayout" layoutstyle="righttoleft" bordercolor="none" dock="fill"
											size="400,400" candragsplitter="true" splitterposition="200,1">
											<div type="custom" cid="latestdiv" name="divLatest"/>
											<div type="custom" cid="mychart" name="mainChart1" cycle="0" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
										</div>
										<div type="splitlayout" layoutstyle="lefttoright" bordercolor="none" dock="fill"
											size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
											<div type="custom" cid="mychart" name="mainChart2" cycle="5" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
											<div type="custom" cid="mychart" name="mainChart3" cycle="1440" candledivpercent="0.7"
													voldivpercent="0.3" backcolor="none" bordercolor="none"/>
										</div>
									</div>
								</div>
							</div>

	                    </div>
	                    </div>
	                    <div type="tabpage" text="多K線" name="divMulti" backcolor="none">
	                        <div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none"
	                            dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="133,1">
	                            <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                    <div type="custom" cid="mychart" name="chart1" cycle="1" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    <div type="custom" cid="mychart" name="chart2" cycle="5" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                                <div type="custom" cid="mychart" name="chart3" cycle="10" nativerefresh="true" candledivpercent="1"
	                                    voldivpercent="0" backcolor="none" bordercolor="none"/>
	                            </div>
	                            <div type="splitlayout" layoutstyle="lefttoright" backcolor="none" bordercolor="none"
	                                dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                    <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                        dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                        <div type="custom" cid="mychart" name="chart4" cycle="15" nativerefresh="true" candledivpercent="1"
	                                            voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                        <div type="custom" cid="mychart" name="chart5" cycle="20" nativerefresh="true" candledivpercent="1"
	                                            voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    </div>
	                                    <div type="custom" cid="mychart" name="chart6" cycle="30" nativerefresh="true" candledivpercent="1"
	                                        voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                                <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                    dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="267,1">
	                                    <div type="splitlayout" layoutstyle="toptobottom" backcolor="none" bordercolor="none"
	                                        dock="fill" size="400,400" candragsplitter="true" splitmode="percentsize" splitterposition="200,1">
	                                        <div type="custom" cid="mychart" name="chart7" cycle="1440" nativerefresh="true"
	                                            candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                        <div type="custom" cid="mychart" name="chart8" cycle="10080" nativerefresh="true"
	                                            candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                    </div>
	                                    <div type="custom" cid="mychart" name="chart9" cycle="43200" nativerefresh="true"
	                                        candledivpercent="1" voldivpercent="0" backcolor="none" bordercolor="none"/>
	                                </div>
	                            </div>
	                        </div>
	                    </div>
	                </div>
	            </div>
	    </body>
	    </html>"""

xml = MAIN_XML
gPaint.render(None, xml)
gridStocks = gPaint.findView("gridStocks")
for i in range(3, len(gridStocks.columns) - 1):
	gridStocks.columns[i].cellAlign = "right"
gridStocks.columns[len(gridStocks.columns) - 1].cellAlign = "center"
if gPaint.defaultUIStyle == "dark":
	gridStocks.selectedRowColor = "rgb(75,75,75)"
	gridStocks.alternateRowColor = "rgb(25,25,25)"
elif gPaint.defaultUIStyle == "light":
	gridStocks.selectedRowColor = "rgb(175,175,175)"
	gridStocks.alternateRowColor = "rgb(245,245,245)"
stockName = "浦發銀行"

progressDiv = gPaint.findView("progress")
stockName = get_stock_profile(currentCode)["name"]
progressDiv.onPaint = drawProgressDiv

temperatureUpButton = gPaint.findView("temperatureUp")
temperatureUpButton.onPrePaint = drawUpButton
temperatureDownButton = gPaint.findView("temperatureDown")
temperatureDownButton.onPrePaint = drawDownButton

topPUpButton = gPaint.findView("topPUp")
topPUpButton.onPrePaint = drawUpButton
topPDownButton = gPaint.findView("topPDown")
topPDownButton.onPrePaint = drawDownButton

temperatureDiv = gPaint.findView("temperatureDiv")
temperatureDiv.onPaint = drawMyDiv
topPDiv = gPaint.findView("topPDiv")
topPDiv.onPaint = drawMyDiv
topPDiv.parent.invalidate()
findViewsByType("mychart", gPaint.views, findMyCharts)
for i in range(0, len(findMyCharts)):
	myChart = findMyCharts[i]
	splitDiv = FCSplitLayoutDiv()
	splitDiv.layoutStyle = "toptobottom"
	splitDiv.size = FCSize(400, 400)
	splitDiv.backColor = "none"
	splitDiv.borderColor = "none"
	splitDiv.dock = "fill"
	myChart.addView(splitDiv)

	topDiv = FCLayoutDiv()
	topDiv.backColor = "none"
	topDiv.borderColor = "none"
	topDiv.layoutStyle = "lefttoright"
	topDiv.showHScrollBar = False

	bottomDiv = FCDiv()
	bottomDiv.backColor = "none"
	bottomDiv.borderColor = "none"

	addViewToSplit(splitDiv, topDiv, bottomDiv, 30)
	if i ==1:
		splitDiv.splitter.location = FCPoint(0, 0)
		
	chart = FCChart()
	chart.leftVScaleWidth = 70
	chart.rightVScaleWidth = 70
	if i > 0:
		chart.leftVScaleWidth = 0
	chart.vScaleDistance = 35
	chart.hScalePixel = 11
	chart.hScaleHeight = 30
	chart.candlePaddingTop = 30
	chart.candlePaddingBottom = 20
	chart.volPaddingTop = 20
	chart.volPaddingBottom = 0
	chart.vScaleDistance = 35
	chart.dock = "fill"
	chart.font = "Default,12"
	chart.candleDivPercent = float(myChart.exAttributes["candledivpercent"])
	chart.volDivPercent = float(myChart.exAttributes["voldivpercent"])
	chart.indDivPercent = 0
	chart.text = currentCode + " " + stockName
	chart.allowDragChartDiv = True
	chart.onPaintChartTip = drawChartTip
	
	charts.append(chart)
	bottomDiv.addView(chart)
	setChartTheme(chart, i)
	# 預測面板屬性調整
	if myChart.viewName=="preDiv":
		chart.viewName = "preChart"
		chart.indicatorColors = []
		chart.indicatorColors.append("rgb(255,255,255)")
		chart.indicatorColors.append("rgb(255,255,0)")
		chart.indicatorColors.append("rgb(150,0,150)")
		chart.indicatorColors.append("rgb(255,0,0)")
		chart.indicatorColors.append("rgb(0,150,150)")
		chart.indicatorColors.append("rgb(0,150,0)")
		chart.indicatorColors.append("rgb(59,174,218)")
		chart.indicatorColors.append("rgb(50,50,50)")
		chart.onPaintChartStock = drawPreChart
		chart.leftVScaleWidth = 0
		chart.mainIndicator = "BOLL"
		chart.showIndicator = "BIAS"
		chart.preMode = "predict"
		chart.lookback = 50
		chart.pred_len = 5
		chart.datas2 = []
		chart.temperature = 1.0
		chart.topP = 0.9
		chart.upColor = "rgb(186,56,18)"
		chart.downColor = "rgb(31,182,177)"
		chart.upColor2 =  "rgb(245,158,132)"
		chart.downColor2 ="rgb(131,199,190)" 
		chart.vScaleTextColor = "rgb(255,255,255)"
	cycles = []
	cycles.append("1")
	cycles.append("5")
	cycles.append("10")
	cycles.append("15")
	cycles.append("20")
	cycles.append("30")
	cycles.append("60")
	cycles.append("90")
	cycles.append("120")
	cycles.append("日")
	cycles.append("周")
	cycles.append("月")
	cycles.append("季")
	cycles.append("半")
	cycles.append("年")
	cyclesInts = []
	cyclesInts.append(1)
	cyclesInts.append(5)
	cyclesInts.append(10)
	cyclesInts.append(15)
	cyclesInts.append(20)
	cyclesInts.append(30)
	cyclesInts.append(60)
	cyclesInts.append(90)
	cyclesInts.append(120)
	cyclesInts.append(1440)
	cyclesInts.append(10080)
	cyclesInts.append(43200)
	cyclesInts.append(129600)
	cyclesInts.append(259200)
	cyclesInts.append(518400)
	for c in range(0, len(cycles)):
		cycleButton = FCButton()
		cycleButton.text = cycles[c]
		cycleButton.size = FCSize(27, 30)
		if gPaint.defaultUIStyle == "dark":
			cycleButton.textColor = "rgb(200,200,200)"
			cycleButton.borderColor = "rgb(50,50,50)"
		elif gPaint.defaultUIStyle == "light":
			cycleButton.textColor = "rgb(50,50,50)"
			cycleButton.borderColor = "rgb(200,200,200)"
		cycleButton.backColor = "none"
		cycleButton.viewName = "cycle," + str(i) + "," + str(cyclesInts[c])
		topDiv.addView(cycleButton)
for myChart in findMyCharts:
	topDiv = myChart.views[0].firstView
	for cycleButton in topDiv.views:
		if cycleButton.viewName.find("cycle,") == 0:
			cycleInt = int(cycleButton.viewName.split(",")[2])
			labelMap = {
				1440: "日",
				10080: "週",
				43200: "月",
				129600: "季",
				259200: "半年",
				518400: "年",
			}
			cycleButton.text = labelMap.get(cycleInt, str(cycleInt))
queryHistoryData(currentCode, stockName, 0, findMyCharts)
queryHistoryData(currentCode, stockName, 1, findMyCharts)
queryHistoryData(currentCode, stockName, 1440, findMyCharts)
queryPrice()
queryNewData()
thread = threading.Thread(target=startQueryNewData, args=())
thread.start()
thread2 = threading.Thread(target=startQueryPriceData, args=())
thread2.start()
gPaint.update()
setWindowSize(FCSize(1400, 900))
setCenterScreen(True)
print("[startup] about to show main window")
showWindow(gPaint)
