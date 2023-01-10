#Instead of starting from scratch we nearly always want to start from some base image. We're going to start from a simple python image
FROM python:3.7-slim


# install essentials in our image
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# copy over our application (the essential parts) from our computer to the container:
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

