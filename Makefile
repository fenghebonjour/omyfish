.PHONY: install train eval app api predict clean

install:
	pip install -r requirements.txt

train:
	python -m src.train --config configs/config.yaml

eval:
	python -m src.evaluate --config configs/config.yaml

app:
	streamlit run app/main.py

api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

predict:
	python -m src.predict --image $(IMAGE)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
