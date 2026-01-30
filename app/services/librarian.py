import json
import os

class Librarian:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = []
        self.load_data()

    def load_data(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"📚 Librarian: Loaded {len(self.data)} Vachanamruts.")
        else:
            print(f"❌ Librarian Error: File not found at {self.json_path}")

    def get_full_text(self, chapter: str, section: str, number: int):
        # Normalize section for comparison (handle "None" string from URL)
        search_section = "" if section == "None" else section

        for item in self.data:
            if (item.get("chapter") == chapter and 
                str(item.get("section")) == str(search_section) and 
                int(item.get("vachanamrut_no")) == number):
                return item
        return None

# Singleton Instance
# We assume the code runs from the 'backend' folder, so path is ./data/...
librarian_service = Librarian(json_path="./data/vachanamrut_cleaned.json")