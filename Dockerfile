FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -u 10001 --create-home workspace && mkdir /data && chown workspace:workspace /data
COPY --chown=workspace:workspace app ./app
COPY --chown=workspace:workspace scripts ./scripts
USER 10001:10001
# Linux host networking, loopback bind, accessed through SSM/VPN. Not a public web deployment.
CMD ["bash", "scripts/start.sh"]
