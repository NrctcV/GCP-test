FROM python:3.7
RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8080
WORKDIR /app
# copy over and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
# copying everything over
COPY . .
# run app
CMD streamlit run app.py