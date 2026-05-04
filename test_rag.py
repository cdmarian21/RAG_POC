import csv
import app
from qa_data import TEST_DATA

def main():
    retriever, llm_chain, section_index = app.setup_pipeline()
    
    if not retriever:
        print("Error with faiss db. Did you run ETL.py?")
        return

    results = []
    total_questions = len(TEST_DATA)

    # looping through the test data and answering each question
    for i, item in enumerate(TEST_DATA, 1):
        question = item["question"]
        expected = item["expected_answer"]
        
        print(f"[{i}/{total_questions}] Question: {question}")
        
        try:
            actual_answer, seen_sections = app.ask_question(question, retriever, llm_chain, section_index)
            print(f"Answered\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
            actual_answer = f"ERROR: {e}"
        
        results.append({
            "ID": i,
            "Question": question,
            "Expected": expected,
            "Model answer": actual_answer.strip()
        })

    # saving to CSV
    csv_file = "rag_evaluation_results.csv"
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["ID", "Question", "Expected", "Model answer"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Eval complete")
        print(f"Results saved to: {csv_file}")

    except Exception as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    main()