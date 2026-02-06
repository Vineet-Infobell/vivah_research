# ============================================================================
# COPY-PASTE THIS INTO YOUR NOTEBOOK TO FIX THE ERRORS
# ============================================================================

# Cell 1: Fixed EmbeddingGenerator Class
# Replace the old EmbeddingGenerator class with this:

class EmbeddingGenerator:
    """Wrapper for Google embedding generation using new GenAI SDK"""
    
    def __init__(self, model_name: str, dimension: int, client=None):
        self.model_name = model_name
        self.dimension = dimension
        self.client = client  # Use the global client
    
    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            # Use the new SDK's client-based API
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension
                )
            )
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return np.zeros(self.dimension)
    
    def generate_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        embeddings = []
        iterator = tqdm(texts, desc=f"Generating embeddings") if show_progress else texts
        
        for text in iterator:
            emb = self.generate_embedding(text)
            embeddings.append(emb)
            time.sleep(0.05)  # Rate limiting
        
        return np.array(embeddings)

print("✅ EmbeddingGenerator class defined (FIXED)")


# ============================================================================
# Cell 2: Enhanced Ranking Function with Profile Display
# ============================================================================

def rank_documents_with_profiles(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    doc_ids: np.ndarray,
    df: pd.DataFrame,
    similarity_fn: Callable,
    top_k: int = 5,
    show_profiles: bool = True
) -> Tuple[List[int], List[Dict]]:
    """
    Rank documents and display profile information with scores
    """
    scores = similarity_fn(query_emb, doc_embs)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_ids = doc_ids[top_indices].tolist()
    top_scores = scores[top_indices].tolist()
    
    # Get profile information
    profile_info = []
    for idx, (user_id, score) in enumerate(zip(top_ids, top_scores), 1):
        profile = df[df['User_ID'] == user_id].iloc[0]
        info = {
            'Rank': idx,
            'User_ID': user_id,
            'Name': profile['Name'],
            'Age': profile['Age'],
            'Gender': profile['Gender'],
            'Job_Title': profile['Job_Title'],
            'Education': profile['Education'],
            'Location': profile['Location'],
            'Religion': profile['Religion'],
            'Score': score
        }
        profile_info.append(info)
    
    if show_profiles:
        print(f"\n{'='*80}")
        print(f"Top {top_k} Profiles:")
        print(f"{'='*80}")
        for info in profile_info:
            print(f"\n{info['Rank']}. {info['Name']} (ID: {info['User_ID']}) - Score: {info['Score']:.4f}")
            print(f"   {info['Age']}y/o {info['Gender']}, {info['Religion']}")
            print(f"   {info['Job_Title']} | {info['Education']}")
            print(f"   📍 {info['Location']}")
    
    return top_ids, profile_info

print("✅ Enhanced ranking function defined")


# ============================================================================
# Cell 3: Updated Benchmark Function
# ============================================================================

def benchmark_single_config(
    model_name: str,
    dimension: int,
    display_name: str,
    profile_strings: List[str],
    user_ids: np.ndarray,
    ground_truth: List[Dict],
    similarity_algorithms: Dict[str, Callable],
    top_k: int = 5
) -> List[Dict]:
    """
    Benchmark a single model+dimension configuration with profile display.
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"🚀 BENCHMARKING: {display_name} (dimension={dimension})")
    print(f"{'='*80}")
    
    # Initialize generator with client (use global 'client' variable)
    try:
        generator = EmbeddingGenerator(model_name, dimension, client=client)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return results
    
    # Generate embeddings for all profiles
    print(f"\n📊 Generating embeddings for {len(profile_strings)} profiles...")
    start_time = time.time()
    
    try:
        profile_embeddings = generator.generate_embeddings_batch(profile_strings, show_progress=True)
        embedding_time = time.time() - start_time
        print(f"✅ Embeddings generated in {embedding_time:.2f}s")
        print(f"   Vector shape: {profile_embeddings.shape}")
        print(f"   Memory size: {profile_embeddings.nbytes / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return results
    
    # Test each similarity algorithm
    for algo_name, similarity_fn in similarity_algorithms.items():
        print(f"\n{'='*80}")
        print(f"🔍 Algorithm: {algo_name}")
        print(f"{'='*80}")
        
        query_metrics = []
        total_search_time = 0.0
        
        # Evaluate on each query
        for gt_idx, gt in enumerate(ground_truth, 1):
            query_text = gt['query']
            relevant_ids = gt['relevant_ids']
            
            print(f"\n\n📝 Query {gt_idx}/{len(ground_truth)}: {query_text[:80]}...")
            
            # Generate query embedding
            try:
                query_emb = generator.generate_embedding(query_text)
            except Exception as e:
                print(f"⚠️ Failed to embed query: {e}")
                continue
            
            # Perform similarity search with profile display
            search_start = time.time()
            retrieved_ids, profile_info = rank_documents_with_profiles(
                query_emb,
                profile_embeddings,
                user_ids,
                df,  # Use global df
                similarity_fn,
                top_k=top_k,
                show_profiles=True  # Show profiles for each query
            )
            search_time = time.time() - search_start
            total_search_time += search_time
            
            # Show which profiles were relevant
            print(f"\n✅ Ground Truth Relevant IDs: {relevant_ids}")
            print(f"🎯 Retrieved IDs: {retrieved_ids}")
            matches = set(retrieved_ids) & set(relevant_ids)
            print(f"✨ Matches: {matches} ({len(matches)}/{len(relevant_ids)})")
            
            # Calculate metrics
            metrics = evaluate_ranking(retrieved_ids, relevant_ids, k=top_k)
            query_metrics.append(metrics)
            
            print(f"\n📊 Metrics: P@5={metrics['Precision@5']:.3f}, R@5={metrics['Recall@5']:.3f}, MRR={metrics['MRR']:.3f}, NDCG={metrics['NDCG']:.3f}")
        
        # Average metrics
        avg_metrics = {
            'Precision@5': np.mean([m['Precision@5'] for m in query_metrics]),
            'Recall@5': np.mean([m['Recall@5'] for m in query_metrics]),
            'MRR': np.mean([m['MRR'] for m in query_metrics]),
            'NDCG': np.mean([m['NDCG'] for m in query_metrics])
        }
        
        avg_search_time = total_search_time / len(ground_truth)
        
        # Store result
        result = {
            'Model': display_name,
            'Embedding_Dimension': dimension,
            'Algorithm': algo_name,
            'Precision@5': avg_metrics['Precision@5'],
            'Recall@5': avg_metrics['Recall@5'],
            'MRR': avg_metrics['MRR'],
            'NDCG': avg_metrics['NDCG'],
            'Embedding_Time_100_Profiles': embedding_time,
            'Search_Time_Per_Query': avg_search_time
        }
        results.append(result)
        
        print(f"\n{'='*80}")
        print(f"📈 AVERAGE METRICS FOR {algo_name}:")
        print(f"{'='*80}")
        print(f"   Precision@5: {avg_metrics['Precision@5']:.3f}")
        print(f"   Recall@5: {avg_metrics['Recall@5']:.3f}")
        print(f"   MRR: {avg_metrics['MRR']:.3f}")
        print(f"   NDCG: {avg_metrics['NDCG']:.3f}")
        print(f"   Avg Search Time: {avg_search_time:.4f}s")
    
    print(f"\n✅ Completed: {display_name} (dim={dimension})")
    return results

print("✅ Benchmark function defined (UPDATED)")


# ============================================================================
# Cell 4: Example Usage
# ============================================================================

# Now run your benchmark like this:
results = benchmark_single_config(
    model_name="models/text-embedding-004",
    dimension=768,
    display_name="Gemini-001",
    profile_strings=profile_strings,
    user_ids=user_ids,
    ground_truth=GROUND_TRUTH,
    similarity_algorithms=SIMILARITY_ALGORITHMS,
    top_k=5
)

# Add to global results
all_results.extend(results)

print(f"\n✅ Total results collected: {len(all_results)}")
