FROM dennisgoldfarb/pytorch_ris:lightning

WORKDIR /workspace/Lodestone
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "lodestone.train", "--config", "config.json"]
