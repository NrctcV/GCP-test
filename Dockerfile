FROM python:3.7
# Copy local code to the container image.
ENV APP_HOME /app

WORKDIR $APP_HOME
COPY . ./

# --------------- Install python packages using `pip` ---------------

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt

# --------------- Configure Streamlit ---------------
RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
	[server]\n\
	enableCORS = false\n\
	" > /root/.streamlit/config.toml'

EXPOSE 8501
EXPOSE 8080

# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["streamlit", "run", "--server.port", "8080", "app.py"]
