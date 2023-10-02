from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
docs = [
    "this is one document",
    "and another document"
]

embeddings = embed_model.embed_documents(docs)

print(f"We have {len(embeddings)} doc embeddings, each with "
      f"a dimensionality of {len(embeddings[0])}.")
import os
import pinecone

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '281413e6-7ba9-4bc8-9792-bfee56a3acc0',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)
import time

index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pinecone.Index(index_name)
index.describe_index_stats()

from google.colab import drive
import pandas as pd
from datasets import Dataset

#drive.mount('/content/drive')
from datasets import load_dataset
path = "/content/sample_data/life_ins_res.csv"
data = pd.read_csv(path)
import csv
import os

input_path = path
output_path = '/content/life_ins_res2.csv'

with open(input_path, 'r') as inp, open(output_path, 'w', newline='') as out:
    reader = csv.reader(inp)
    writer = csv.writer(out, delimiter=',')

    # Check if there are any rows left to read
    try:
        header = next(reader)
    except StopIteration:
        header = []

    # Write the header with the "ID" column added
    writer.writerow(['ID'] + header)

    # Write the rows with row numbers
    writer.writerows([i] + row for i, row in enumerate(reader, 1))