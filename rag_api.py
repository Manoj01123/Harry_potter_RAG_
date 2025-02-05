# from fastapi import FastAPI, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.requests import Request
# from rag_retrieval import query_rag
# import uvicorn
# import os
# import requests
#
# # Initialize FastAPI app
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
#
# # Helper function to download files from S3
# def download_from_s3(url, local_path):
#     response = requests.get(url, stream=True)
#     with open(local_path, "wb") as file:
#         for chunk in response.iter_content(chunk_size=8192):
#             file.write(chunk)
#
# @app.get("/", response_class=HTMLResponse)
# def get_home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "answer": None})
#
# # @app.post("/query", response_class=HTMLResponse)
# # def query_rag_endpoint(request: Request, question: str = Form(...)):
# #     # S3 file URL
# #     s3_url = "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf"
# #     local_path = "model.gguf"
# #
# #     # Download file if not already present
# #     if not os.path.exists(local_path):
# #         download_from_s3(s3_url, local_path)
# #
# #     response = query_rag(question)
# #     return templates.TemplateResponse(
# #         "index.html",
# #         {"request": request, "answer": response["result"], "question": question}
# #     )
# #
# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8000)
#
#
# @app.post("/query", response_class=HTMLResponse)
# def query_rag_endpoint(request: Request, question: str = Form(...)):
#     # S3 URLs for both files
#     s3_urls = {
#         "Llama-3.2-3B-Instruct-Q4_0.gguf": "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf",
#         "model.gguf": "https://harrypotter07.s3.us-east-1.amazonaws.com/model.gguf"
#     }
#
#     # Download each file if not already present
#     for filename, s3_url in s3_urls.items():
#         local_path = f"models/{filename}"
#         if not os.path.exists(local_path):
#             download_from_s3(s3_url, local_path)
#
#     # Use the model to generate the response
#     response = query_rag(question)
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request, "answer": response["result"], "question": question}
#     )
#



from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from rag_retrieval import query_rag
import uvicorn
import os
import requests
import logging

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Helper function to download files from S3
# def download_from_s3(url, local_path):
#     logging.info(f"Downloading model from S3: {url}")
#     response = requests.get(url, stream=True)
#     with open(local_path, "wb") as file:
#         for chunk in response.iter_content(chunk_size=8192):
#             file.write(chunk)
#     logging.info(f"Model downloaded and saved to {local_path}")

def download_from_s3(url, local_path):
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping download.")
        return

    try:
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError if the request returned an unsuccessful status code
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded model to {local_path}.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise


# Load model once globally
model_initialized = False
local_path = "model.gguf"
s3_url = "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf"

if not os.path.exists(local_path):
    download_from_s3(s3_url, local_path)

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/query", response_class=HTMLResponse)
def query_rag_endpoint(request: Request, question: str = Form(...)):
    global model_initialized
    if not model_initialized:
        logging.info("Initializing RAG model...")
        model_initialized = True

    response = query_rag(question)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": response["result"], "question": question}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
