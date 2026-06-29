FROM python:3.11-slim

WORKDIR /app
COPY medium.py .

# No extra dependencies — stdlib only
EXPOSE 9000/udp

CMD ["python", "-u", "medium.py"]
