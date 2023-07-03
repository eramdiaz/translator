FROM python:3.10.6-slim

RUN adduser --system --group app \
	&& apt update \
	&& apt install -y git \
	&& git clone https://github.com/eramdiaz/translator.git \
	&& chown -R app:app /translator

WORKDIR /translator

RUN pip install -r requirements.txt \
	&& pip install -r requirements-test.txt

USER app

CMD ["python3"]
