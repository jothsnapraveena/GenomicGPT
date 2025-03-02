# %%
#from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import pickle

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import json
import os
os.environ["USER_AGENT"] = "agent123"

# %% [markdown]
# ### AgenticRAG

# %%
def fetch_ncbi_bookshelf_articles(gene, disease):
    """Fetches review articles for a gene-disease pair from NCBI Bookshelf."""
    
    search_query = f"{gene} {disease} review"
    url = f"https://www.ncbi.nlm.nih.gov/books/?term={search_query.replace(' ', '+')}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "NCBI Bookshelf request failed"}

    soup = BeautifulSoup(response.text, "html.parser")

    review_articles = []
    for rev in soup.find_all("p", class_="title")[:5]:  # ✅ Fetch top 5 reviews
        review_title = rev.text.strip()
        
        # ✅ Fetch NCBI Bookshelf URL (Relative -> Absolute)
        article_link = rev.find("a")["href"]
        full_url = f"https://www.ncbi.nlm.nih.gov{article_link}"
        
        # ✅ Store metadata (Title + URL)
        review_articles.append({
            "title": review_title,
            "url": full_url
        })

    return {"review_articles": review_articles}

'''
# ✅ Test Function
gene = "BRCA1"
disease = "Breast Cancer"
articles = fetch_ncbi_bookshelf_articles(gene, disease)

# ✅ Print Retrieved Articles
for idx, article in enumerate(articles["review_articles"], 1):
    print(f"{idx}. {article['title']}")
    print(f"   URL: {article['url']}")'
'''


# %%
# from scholarly import scholarly
# import requests
# from bs4 import BeautifulSoup

# def get_citations_from_scholar(article_title):
#     """Fetch citation count for a review article using Google Scholar."""
#     try:
#         search_query = scholarly.search_pubs(article_title)
#         result = next(search_query, None)  # Get the first result
        
        
#         if result:
#             return result["num_citations"]  # Extract the citation count
#         else:
#             return 0  # No citations found
#     except Exception as e:
#         print(f"❌ Error fetching citations for {article_title}: {e}")
#         return 0


# %%
def get_citations(article_title):
    """Fetch citation count for a review article using multiple sources (CrossRef, OpenAlex, Semantic Scholar)."""

    # ✅ Try CrossRef first
    citations = get_citations_from_crossref(article_title)
    if citations > 0:
        return citations


# ✅ Fetch Citations via CrossRef API
def get_citations_from_crossref(article_title):
    """Fetches citation count for an article from CrossRef API."""
    url = f"https://api.crossref.org/works?query.title={article_title}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        for item in data["message"]["items"]:
            if "is-referenced-by-count" in item:
                return item["is-referenced-by-count"]

        return 0  # No citations found

    except Exception as e:
        print(f"❌ Error fetching citations from CrossRef for {article_title}: {e}")
        return 0

# %%
def fetch_ncbi_bookshelf_articles(gene, disease):
    """Fetches review articles for a gene-disease pair from NCBI Bookshelf and ranks them by citations."""
    
    search_query = f"{gene} {disease} review"
    url = f"https://www.ncbi.nlm.nih.gov/books/?term={search_query.replace(' ', '+')}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "NCBI Bookshelf request failed"}

    soup = BeautifulSoup(response.text, "html.parser")

    review_articles = []
    for rev in soup.find_all("p", class_="title")[:5]:  # ✅ Fetch top 5 reviews
        review_title = rev.text.strip()
        
        # ✅ Fetch NCBI Bookshelf URL (Relative -> Absolute)
        article_link = rev.find("a")["href"]
        full_url = f"https://www.ncbi.nlm.nih.gov{article_link}"
        
        # ✅ Fetch citation count from Google Scholar
        citations = get_citations_from_crossref(review_title)
        
        # ✅ Store metadata (Title + URL + Citations)
        review_articles.append({
            "title": review_title,
            "url": full_url,
            "citations": citations
        })

    # ✅ Sort articles by number of citations (Descending)
    review_articles.sort(key=lambda x: x["citations"], reverse=True)

    return {"review_articles": review_articles}


# %% [markdown]
# ### RAG for top 5 cited articles

# %%
def load_articles_with_webloader(review_articles):
    """Loads full text from review article URLs using LangChain WebLoader."""
    docs = []
    
    print_outs = []
    for article in review_articles:
        url = article["url"]
        print_outs.append(f"🌐 Scraping: {url}")
        
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            #print("loader", loader.load())
        except Exception as e:
            print(f"❌ Error loading {url}: {e}")
    
    return docs, print_outs  # ✅ Returns extracted documents

# %%
def split_articles(docs):
    """Splits loaded documents into smaller chunks for efficient storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # ✅ Each chunk = 1000 characters
        chunk_overlap=200  # ✅ Allow 200 character overlap
    )
    
    return text_splitter.split_documents(docs)

# %%
def store_in_chromadb(chunks):
    """Stores document chunks in ChromaDB using OpenAI embeddings."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore  # ✅ Returns stored vector index

# %%
def retrieve_relevant_docs(vectorstore, query, top_k=3):
    """Retrieves most relevant documents from ChromaDB for a given query."""
    return vectorstore.similarity_search(query, k=top_k)

# %%
# import openai
# import json

# def predict_mechanism_with_rag(gene, disease, retrieved_docs):
#     """Predicts disease mechanism using retrieved literature and GPT-4."""
    
#     evidence_text = ""
#     for doc in retrieved_docs:
#         evidence_text += f"\n\n{doc.page_content[:2000]}"  # ✅ Limit to 2000 chars

#     # ✅ Prompt GPT-4 with retrieved evidence
#     prompt = f"""
#     Based on the following literature, determine the mechanism of disease for {gene} in {disease}:

#     {evidence_text}

#     Classify as:
#     1. Loss of Function (LoF) - Mutation leads to reduced/absent protein function.
#     2. Gain of Function (GoF) - Mutation leads to enhanced/new protein function.
#     3. Dominant Negative (DN) - Mutant protein interferes with wild-type protein.

#     Response format (JSON):
#     {
#         "mechanism": "...",
#         "justification": "..."
#     }
#     """
    

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content


# %%
#import openai

# ✅ Initialize OpenAI client
#client = openai.OpenAI()

def predict_mechanism_with_rag(gene, disease, retrieved_docs):
    """Predicts disease mechanism using retrieved literature and GPT-4 with enforced JSON output."""

    # ✅ Combine evidence from retrieved documents
    evidence_text = "\n\n".join([doc.page_content[:2000] for doc in retrieved_docs])  # Limit text size

    # ✅ Construct structured prompt
    prompt = f"""
    Based on the following scientific literature, determine the mechanism of disease for {gene} in {disease}:

    {evidence_text}

    Choose one of the following classifications:
    - Loss of Function (LoF): Mutation leads to reduced/absent protein function.
    - Gain of Function (GoF): Mutation leads to enhanced/new protein function.
    - Dominant Negative (DN): Mutant protein interferes with wild-type protein.

    Return a **valid JSON response** in this exact format:
    {{
        "mechanism": "Loss of Function (LoF) | Gain of Function (GoF) | Dominant Negative (DN)",
        "justification": "A concise explanation based on extracted evidence."
    }}
    """

    try:
        # ✅ Enforce JSON output
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}  # ✅ Forces GPT-4 to return JSON
        )

        # ✅ Parse and return JSON response
        return json.loads(response.choices[0].message.content)  # Ensure valid JSON parsing

    except Exception as e:
        print(f"❌ Error predicting mechanism: {e}")
        return {"error": "Failed to predict mechanism"}


# %%

def evaluate_justification_strength(justification_text):
    """Uses GPT-4 to rate the quality of justification on a scale of 0 to 3."""
    prompt = f"""
    Evaluate the following justification for a disease mechanism and return a score between 0 and 3:
    
    {justification_text}

    Score Guide:
    - 0: No clear reasoning
    - 1: Weak or vague reasoning
    - 2: Moderate strength, some references
    - 3: Strong, well-supported by evidence

    Return JSON:
    {{
        "justification_score": ...
    }}
    """

    client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)["justification_score"]


def score_confidence_with_rag(gene, disease, review_articles, mechanism_result, retrieved_docs):
    """
    Assigns a confidence score based on citations, consistency, and justification quality.
    """

    # ✅ 1. Extract relevant metrics
    total_citations = sum([article["citations"] for article in review_articles])
    avg_citations = total_citations / max(len(review_articles), 1)  # Prevent division by zero
    
    # ✅ 2. Analyze consistency across sources
    mechanism_counts = {"LoF": 0, "GoF": 0, "DN": 0}
    
    for doc in retrieved_docs:
        text = doc.page_content.lower()
        if "loss of function" in text or "lof" in text:
            mechanism_counts["LoF"] += 1
        if "gain of function" in text or "gof" in text:
            mechanism_counts["GoF"] += 1
        if "dominant negative" in text or "dn" in text:
            mechanism_counts["DN"] += 1

    most_common_mechanism = max(mechanism_counts, key=mechanism_counts.get)
    consistency_score = mechanism_counts[most_common_mechanism] / max(len(retrieved_docs), 1)

    # ✅ 3. Use GPT-4 to evaluate justification strength
    justification_strength = evaluate_justification_strength(mechanism_result["justification"])

    # ✅ 4. Compute final confidence score (Weighted Average)
    confidence_score = np.clip((
        (avg_citations / 500) * 2 +  # Normalized citation impact (out of 2)
        (consistency_score * 5) +    # Consistency across sources (out of 5)
        (justification_strength * 3) # Justification quality (out of 3)
    ), 1, 10)  # Ensure the score is between 1-10

    return {
        "mechanism": mechanism_result["mechanism"],
        "confidence_score": round(confidence_score, 1),  # Keep decimals
        "consistency": round(consistency_score, 2),
        "justification_strength": round(justification_strength, 2),
        "avg_citations": round(avg_citations, 2)
    }


# %%
def autonomous_mechanism_discovery(gene, disease):
    try:
        with open("data.pkl", "rb") as file:
            final_report, print_outs_final = pickle.load(file)  # Unpacking tuple
        return final_report, print_outs_final
    except Exception as e:
        print("Error loading pickle file:", e)

    """
    Runs a fully autonomous pipeline for discovering the mechanism of disease
    using a combination of web scraping, vector search, RAG, and LLM-based reasoning.
    """
    print_outs_final = []
    print_outs = []
    # 🔍 *Step 1: Fetch Highly Cited Review Articles*
    print_outs.append("\n🔍 Step 1: Searching for Review Articles...")
    print_outs.append(f"🔎 Searching NCBI Bookshelf for highly cited review articles on {gene} and {disease}...")

    search_results = fetch_ncbi_bookshelf_articles(gene, disease)
    review_articles = search_results["review_articles"]

    if not review_articles:
        print_outs.append(f"⚠️ No review articles found for {gene} and {disease}. Exiting process.")
        return {"error": "No review articles found"}

    print_outs.append(f"✅ Found {len(review_articles)} relevant review articles. Selecting the top cited ones.")
    for article in review_articles[:5]:
        print_outs.append(f"📖 {article['title']} (Citations: {article['citations']}) - {article['url']}")
    print_outs_final.append(print_outs.copy())
    print_outs = []
    # 🌐 *Step 2: Scraping Full-Text Articles Using LangChain WebLoader*
    print_outs.append("\n🌐 Step 2: Scraping Articles Using LangChain WebLoader...")
    print_outs.append(f"🔄 Extracting full-text content from the top 5 cited articles...")

    docs, print_outstmp = load_articles_with_webloader(review_articles[:5])
    print_outs.extend(print_outstmp)

    if not docs:
        print_outs.append("⚠️ Failed to extract content from the articles. Exiting process.")
        return {"error": "Web scraping failed"}

    print_outs.append(f"✅ Successfully scraped {len(docs)} documents.")
    print_outs_final.append(print_outs.copy())
    print_outs = []

    # 📄 *Step 3: Splitting Articles into Chunks for Storage & Retrieval*
    print_outs.append("\n📄 Step 3: Splitting Articles into Chunks...")
    print_outs.append("🔹 Long texts need to be split into smaller segments to allow efficient search & retrieval.")

    chunks = split_articles(docs)

    if not chunks:
        print_outs.append("⚠️ No chunks were created from the documents. Exiting process.")
        return {"error": "Text splitting failed"}

    print_outs.append(f"✅ Created {len(chunks)} text chunks from the extracted documents.")
    print_outs_final.append(print_outs.copy())
    print_outs = []

    # 🗄️ *Step 4: Storing Processed Chunks into ChromaDB (Vector Store)*
    print_outs.append("\n🗄️ Step 4: Storing in ChromaDB...")
    print_outs.append("🔹 Storing processed document chunks into ChromaDB for efficient similarity search.")

    vectorstore = store_in_chromadb(chunks)

    if not vectorstore:
        print_outs.append("⚠️ Failed to store chunks in ChromaDB. Exiting process.")
        return {"error": "ChromaDB storage failed"}

    print_outs.append("✅ Successfully stored document chunks in ChromaDB.")
    print_outs_final.append(print_outs.copy())
    print_outs = []

    # 🔎 *Step 5: Retrieving Top-K Relevant Chunks from ChromaDB*
    print_outs.append("\n🔎 Step 5: Retrieving Top Relevant Evidence...")
    print_outs.append(f"🔹 Searching for the most relevant text chunks that discuss {gene} and {disease} mechanism.")

    retrieved_docs = retrieve_relevant_docs(vectorstore, f"{gene} {disease} mechanism", top_k=3)

    if not retrieved_docs:
        print_outs.append("⚠️ No relevant evidence found in ChromaDB. Exiting process.")
        return {"error": "No relevant evidence retrieved"}

    print_outs.append(f"✅ Retrieved {len(retrieved_docs)} relevant document chunks.")
    for i, doc in enumerate(retrieved_docs, 1):
        val = doc.page_content[0:doc.page_content.index("\n")]
        print_outs.append(f'📌 Evidence {i}: {val}...')  # Preview first 250 characters
    print_outs_final.append(print_outs.copy())
    print_outs = []

    # 🧠 *Step 6: Predicting Mechanism of Disease Using RAG & GPT-4*
    print_outs.append("\n🧠 Step 6: Predicting Mechanism Using RAG & GPT-4...")
    print_outs.append("🔹 Feeding retrieved evidence into GPT-4 to classify the mechanism of disease.")

    mechanism_result = predict_mechanism_with_rag(gene, disease, retrieved_docs)

    if "mechanism" not in mechanism_result:
        print_outs.append("⚠️ Mechanism prediction failed. Exiting process.")
        return {"error": "Mechanism prediction failed"}

    print_outs.append(f"✅ Predicted Mechanism: {mechanism_result['mechanism']}")
    print_outs.append(f"📜 Justification: {mechanism_result['justification']}")
    print_outs_final.append(print_outs.copy())
    print_outs = []

    # 📊 *Step 7: Scoring Confidence Based on Citations & Justification Strength*
    print_outs.append("\n📊 Step 7: Scoring Confidence Based on Citations & Evidence...")
    print_outs.append("🔹 Assigning confidence score based on citations, justification quality, and consistency.")

    confidence_score = score_confidence_with_rag(gene, disease, review_articles, mechanism_result, retrieved_docs)

    if "confidence_score" not in confidence_score:
        print_outs.append("⚠️ Confidence scoring failed. Exiting process.")
        return {"error": "Confidence scoring failed"}

    print_outs.append(f"✅ Final Confidence Score: {confidence_score['confidence_score']} (Scale: 1-10)")
    print_outs_final.append(print_outs.copy())
    
    print(print_outs_final)
    print(f"📈 Consistency Score: {confidence_score['consistency']} | Justification Strength: {confidence_score['justification_strength']} | Avg. Citations: {confidence_score['avg_citations']}")

    # ✅ *Final Mechanism Report*
    final_report = {
        "gene": gene,
        "disease": disease,
        "steps": {
            "Step 1": f"Fetched {len(review_articles)} highly cited review articles from NCBI Bookshelf.",
            "Step 2": f"Scraped full text from {len(docs)} top cited articles using WebLoader.",
            "Step 3": f"Split text into {len(chunks)} small chunks for efficient retrieval.",
            "Step 4": "Stored extracted text chunks into ChromaDB vector database.",
            "Step 5": f"Retrieved {len(retrieved_docs)} most relevant evidence snippets from ChromaDB.",
            "Step 6": f"GPT-4 predicted the mechanism of disease as {mechanism_result['mechanism']}.",
            "Step 7": f"Assigned a confidence score of {confidence_score['confidence_score']} based on evidence strength."
        },
        "mechanism": mechanism_result["mechanism"],
        "confidence_score": confidence_score["confidence_score"],
        "justification": mechanism_result["justification"]
    }

    print("\n✅ FINAL MECHANISM REPORT:")
    print(json.dumps(final_report, indent=4))

    with open("data.pkl", "wb") as file:
        pickle.dump((final_report, print_outs_final), file)

    return final_report, print_outs_final

# %%
if __name__ == "__main__":
    # ✅ Run the pipeline
    review_articles, final_report = autonomous_mechanism_discovery("BRCA1", "Breast Cancer")

# %%

load_articles_with_webloader([{
            "title": "Hello",
            "url": "https://www.ncbi.nlm.nih.gov/books/NBK82221/?term=BRCA1%20Breast%20Cancer%20review",
            "citations": 103
        }])

