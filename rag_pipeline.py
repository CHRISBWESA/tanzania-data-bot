# rag_pipeline.py
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
from dotenv import load_dotenv
import re
from typing import Tuple, List, Dict, Any

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Globals to be initialized lazily
DATA = None
REGIONS = None
REGION_EMBEDDINGS = None
FLAT_STATS = None
VECTORIZER = None
TFIDF_MATRIX = None
_initialized = False

def initialize(force: bool = False):
    global DATA, REGIONS, REGION_EMBEDDINGS, FLAT_STATS, VECTORIZER, TFIDF_MATRIX, _initialized

    if _initialized and not force:
        return

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Load data
    file_path = "nbs.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Adjust for the nested structure
    DATA = raw_data.get("tanzania_census_2022", {})
    if not DATA or "regions" not in DATA:
        raise ValueError("No 'tanzania_census_2022' or 'regions' key found in nbs.json")
    
    REGIONS = [region["name"] for region in DATA.get("regions", [])]
    if not REGIONS:
        raise ValueError("No regions found in nbs.json. Ensure 'regions' contains at least one region.")
    
    REGION_EMBEDDINGS = {region: get_embedding(region.lower()) for region in REGIONS}
    
    FLAT_STATS = []
    for region_data in DATA.get("regions", []):
        region = region_data["name"]
        for stat_category in ["population", "buildings", "health_facilities", "schools"]:
            stats = region_data.get(stat_category, {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    text = f"{region} {stat_category} {key.replace('_', ' ')} {value}"
                    FLAT_STATS.append({
                        "region": region,
                        "stat_category": stat_category,
                        "stat_key": key,
                        "value": value,
                        "text": text
                    })
    
    if not FLAT_STATS:
        raise ValueError("No valid statistics found in nbs.json. Ensure regions have population, buildings, etc.")
    
    STAT_TEXTS = [stat["text"] for stat in FLAT_STATS]
    
    if not any(STAT_TEXTS):
        raise ValueError("All statistic texts are empty. Check the data values in nbs.json.")
    
    try:
        VECTORIZER = TfidfVectorizer(stop_words='english', min_df=1)
        TFIDF_MATRIX = VECTORIZER.fit_transform(STAT_TEXTS)
        if len(VECTORIZER.vocabulary_) == 0:
            raise ValueError("Empty vocabulary after vectorization. Ensure region names and stat keys are meaningful text.")
    except ValueError as ve:
        raise ValueError(f"Vectorization failed: {str(ve)}. Ensure nbs.json has sufficient textual content.")
    
    _initialized = True

def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding for text using Gemini.
    """
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        return np.array(result['embedding'])
    except Exception as e:
        raise ValueError(f"Failed to get embedding for '{text}': {str(e)}")

def detect_language(query: str) -> str:
    """
    Simple language detection: if contains Swahili words like 'ya', 'katika', assume Kiswahili.
    """
    swahili_keywords = ["ya", "katika", "ni", "kwa", "na", "idadi", "watu", "mkoa"]
    if any(word in query.lower() for word in swahili_keywords):
        return "swahili"
    return "english"

def cosine_sim_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform cosine similarity search using embeddings for regions and TF-IDF for stats.
    """
    query_lower = query.lower()
    
    # Region matching using embeddings
    query_emb = get_embedding(query_lower)
    sim_scores = {region: cosine_similarity([query_emb], [emb])[0][0] for region, emb in REGION_EMBEDDINGS.items()}
    sorted_regions = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Stat matching using TF-IDF
    query_vec = VECTORIZER.transform([query_lower])
    tfidf_sims = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()
    top_indices = np.argsort(tfidf_sims)[-top_k:]
    
    # Combine results
    relevant_chunks = []
    for region, score in sorted_regions:
        if score > 0.5:  # Threshold
            relevant_chunks.append({"region": region, "score": score})
    
    for idx in top_indices:
        if tfidf_sims[idx] > 0.1:
            relevant_chunks.append(FLAT_STATS[idx])
    
    return relevant_chunks[:top_k]

def agentic_rag(query: str) -> List[Dict]:
    """
    Agentic RAG: Use Gemini to decide what chunks to select.
    Fallback to cosine sim.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
        Given the user query: "{query}"
        And available regions: {', '.join(REGIONS)}
        And stats: population (total, male, female), buildings (total, multi_storey, single_storey, under_construction, with_physical_address, without_physical_address), health_facilities (dispensary, health_centre, hospital), schools (primary, secondary)
        
        Decide which regions and stats are relevant.
        Output as JSON list of dicts like [{"region": "Dar es Salaam", "stat_category": "population", "stat_key": "total"}]
        """
        response = model.generate_content(prompt)
        chunks = json.loads(response.text)
        return chunks
    except Exception as e:
        print(f"Gemini RAG failed: {str(e)}. Falling back to cosine similarity.")
        return cosine_sim_search(query)

def perform_calculations(chunks: List[Dict], query: str) -> Dict:
    """
    Perform numeric operations: sum, diff, ratio, comparison, ranking.
    """
    # Find region data in DATA["regions"]
    region_dict = {r["name"]: r for r in DATA["regions"]}
    
    # Extract values
    values = {}
    for chunk in chunks:
        region = chunk.get("region")
        stat_category = chunk.get("stat_category", "population")
        stat_key = chunk.get("stat_key", "total")
        if region in region_dict and stat_category in region_dict[region] and stat_key in region_dict[region][stat_category]:
            value = region_dict[region][stat_category][stat_key]
            values[f"{region}_{stat_category}_{stat_key}"] = value
    
    # Detect operation
    query_lower = query.lower()
    if "top" in query_lower and "by" in query_lower:
        # Ranking
        n = int(re.search(r'top (\d+)', query_lower).group(1)) if re.search(r'top (\d+)', query_lower) else 5
        stat_match = re.search(r'by (\w+)', query_lower)
        stat_category = stat_match.group(1) if stat_match and stat_match.group(1) in ["population", "buildings", "health_facilities", "schools"] else "population"
        stat_key = "total"
        
        # Special handling for schools (sum primary + secondary)
        if stat_category == "schools":
            all_values = []
            for r in REGIONS:
                primary = region_dict[r].get("schools", {}).get("primary", 0)
                secondary = region_dict[r].get("schools", {}).get("secondary", 0)
                total_schools = primary + secondary
                all_values.append((r, total_schools))
        else:
            stat_key_match = re.search(r'by (\w+ \w+)', query_lower)
            stat_key = stat_key_match.group(1).replace(" ", "_") if stat_key_match else "total"
            all_values = [(r, region_dict[r].get(stat_category, {}).get(stat_key, 0)) for r in REGIONS]
        
        sorted_values = sorted(all_values, key=lambda x: x[1], reverse=True)[:n]
        return {"type": "ranking", "data": sorted_values}
    
    elif "list all" in query_lower or "all regions" in query_lower:
        # Listing
        stat_match = re.search(r'(population|buildings|health_facilities|schools)', query_lower)
        stat_category = stat_match.group(1) if stat_match else "population"
        stat_key = "total"
        if stat_category == "schools":
            all_values = [(r, region_dict[r].get("schools", {}).get("primary", 0) + region_dict[r].get("schools", {}).get("secondary", 0)) for r in REGIONS]
        else:
            stat_key_match = re.search(r'(total|multi_storey|single_storey|under_construction|with_physical_address|without_physical_address|dispensary|health_centre|hospital|primary|secondary)', query_lower)
            stat_key = stat_key_match.group(1) if stat_key_match else "total"
            all_values = [(r, region_dict[r].get(stat_category, {}).get(stat_key, 0)) for r in REGIONS]
        return {"type": "listing", "data": all_values}
    
    elif "add" in query_lower or "sum" in query_lower:
        # Summation
        total = sum(v for v in values.values())
        return {"type": "sum", "data": total}
    
    elif "difference" in query_lower or "subtract" in query_lower:
        # Difference
        if len(values) == 2:
            vals = list(values.values())
            diff = abs(vals[0] - vals[1])
            return {"type": "difference", "data": diff}
    
    elif "ratio" in query_lower:
        # Ratio
        if len(values) == 2:
            vals = list(values.values())
            ratio = vals[0] / vals[1] if vals[1] != 0 else 0
            return {"type": "ratio", "data": ratio}
    
    elif "larger" in query_lower or "greater" in query_lower or "compare" in query_lower:
        # Comparison
        if len(values) >= 2:
            max_key = max(values, key=values.get)
            return {"type": "comparison", "data": max_key.split("_")[0]}
    
    else:
        # Default: return values
        return {"type": "values", "data": values}

def generate_response(query: str, chunks: List[Dict], calc_result: Dict) -> str:
    """
    Use Gemini to generate natural language response.
    """
    lang = detect_language(query)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    prompt = f"""
    User query: "{query}"
    Relevant data: {json.dumps(chunks)}
    Calculations: {json.dumps(calc_result)}
    
    Generate a short answer (2-5 sentences) using the data.
    Include numbers with units and year 2022.
    If query in Swahili, respond in Swahili.
    Format nicely with bullets or tables if needed.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini response generation failed: {str(e)}")
        # Fallback response
        if calc_result["type"] == "values":
            return "\n".join([f"{k.replace('_', ' ')}: {v} (2022)" for k, v in calc_result["data"].items()])
        elif calc_result["type"] == "ranking":
            return "\n".join([f"- {r}: {v:,} (2022)" for r, v in calc_result["data"]])
        elif calc_result["type"] == "listing":
            return "\n".join([f"- {r}: {v:,} (2022)" for r, v in calc_result["data"]])
        elif calc_result["type"] == "sum":
            return f"Total: {calc_result['data']:,} (2022)"
        elif calc_result["type"] == "difference":
            return f"Difference: {calc_result['data']:,} (2022)"
        elif calc_result["type"] == "ratio":
            return f"Ratio: {calc_result['data']:.2f} (2022)"
        elif calc_result["type"] == "comparison":
            return f"{calc_result['data']} has the larger value (2022)"
        return "Unable to generate response due to an error."

def handle_greeting(query: str) -> str:
    """
    Handle greetings.
    """
    greetings_en = ["hello", "hi", "hey"]
    greetings_sw = ["mambo", "vipi", "habari", "salamu"]
    query_lower = query.lower()
    if any(g in query_lower for g in greetings_en):
        return "Hello! How can I help with Tanzania Census 2022 data?"
    elif any(g in query_lower for g in greetings_sw):
        return "Habari! Ninawezaje kukusaidia kuhusu data ya Sensa ya Watu na Makazi 2022 Tanzania?"
    return None

def process_query(query: str) -> Tuple[str, Any]:
    """
    Main function to process user query.
    """
    global _initialized
    if not _initialized:
        initialize()
    
    if not query.strip():
        raise ValueError("Empty query")
    
    greeting_resp = handle_greeting(query)
    if greeting_resp:
        return greeting_resp, None
    
    # Check off-topic
    census_keywords = ["population", "people", "watu", "buildings", "majengo", "health", "afya", "schools", "shule", "region", "mkoa", "census", "sensa"]
    if not any(k in query.lower() for k in census_keywords):
        return "Sorry, I only answer questions about the Tanzania Population & Housing Census 2022.", None
    
    # Agentic RAG retrieval
    chunks = agentic_rag(query)
    
    if not chunks:
        return "No matching data found in the census.", None
    
    # Perform calculations
    calc_result = perform_calculations(chunks, query)
    
    # Generate response
    response = generate_response(query, chunks, calc_result)
    
    return response, chunks  # Sources are the chunks