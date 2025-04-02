FROM python:3.9

WORKDIR /TradePredictor

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["pytest"]
