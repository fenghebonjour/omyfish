.PHONY: install train resume eval app api predict clean db compose-up compose-down download-na-freshwater

install:
	pip install -r requirements.txt

train:
	python -m services.fish_ai.training.train --config configs/training.yaml

resume:
	python -m services.fish_ai.training.train --config configs/training.yaml --resume

eval:
	python -m services.fish_ai.training.evaluate --config configs/training.yaml

app:
	streamlit run apps/omyfish_web/main.py

api:
	uvicorn apps.omyfish_api.main:app --reload --host 0.0.0.0 --port 8000

predict:
	python -m services.fish_ai.predictors.efficientnet --image $(IMAGE)

db:
	docker compose -f infrastructure/docker/docker-compose.yml up postgis -d

compose-up:
	docker compose -f infrastructure/docker/docker-compose.yml up --build -d

compose-down:
	docker compose -f infrastructure/docker/docker-compose.yml down

download-na-freshwater:
	python research/scripts/download_data.py download-na-freshwater --count $(or $(COUNT),400)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
