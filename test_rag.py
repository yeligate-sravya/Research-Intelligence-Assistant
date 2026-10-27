


from rag_pipeline import retrieve_chunks

def main():
    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("ERROR: Please enter a question.")
        return

    try:
        chunks = retrieve_chunks(question, top_k=3)

        if not chunks:
            print("\nFAIL: No chunks retrieved.")
            return

        print(f"\nPASS: Retrieved {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Result {i} ---")
            print("Source:", chunk.get("source"))
            print("Page:", chunk.get("page"))
            print("Chunk ID:", chunk.get("chunk_id"))
            print("Distance:", chunk.get("distance"))
            print("Text:", chunk.get("text", "")[:500])

    except Exception as e:
        print("\nFAIL: Retrieval test encountered an error.")
        print(type(e).__name__ + ":", e)

if __name__ == "__main__":
    main()
