FROM python:3.7
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# See:  https://stackoverflow.com/questions/59052104/how-do-you-deploy-a-streamlit-app-on-app-engine-gcp
FROM python:3.7
EXPOSE 8080
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# --------------- Install python packages using `pip` ---------------

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt

# Run the web service on container startup.
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
