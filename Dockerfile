FROM python:3.7

FROM python:3.7
EXPOSE 8080
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt


CMD streamlit run --server.port 8080 --server.enableCORS false app.py
