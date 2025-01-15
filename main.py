import pandas as pd
from bs4 import BeautifulSoup
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from difflib import get_close_matches
import difflib
import json


class NetworkKeywordExtractor:
    def __init__(self):
        # Initialize NLTK components
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")

        # Initialize stop words and technical terms
        self.stop_words = self._initialize_stop_words()
        self.technical_terms = self._initialize_technical_terms()

        # Network tracking
        self.tag_relationships = defaultdict(set)  # Track which tags appear together
        self.tag_frequency = Counter()  # Track how often each tag is used
        self.existing_tags = set()  # Store all tags we've seen

    def _initialize_stop_words(self):
        """Initialize enhanced stop words with merged functionality."""
        try:
            base_stop_words = set(stopwords.words("english"))
        except:
            base_stop_words = set()

        # Common word stems that will have variations
        word_stems = {
            # Domain-specific terms that are too common
            "resource",
            "file",
            "document",
            "page",
            "article",
            "post",
            "message",
            "content",
            "data",
            "list",
            "item",
            "note",
            "example",
            "reference",
            "guide",
            "tutorial",
            "project",
            "tool",
            "video",
            "image",
            "link",
            # Technical process words
            "create",
            "update",
            "delete",
            "remove",
            "add",
            "edit",
            "change",
            "modify",
            "view",
            "show",
            "display",
            "hide",
            "find",
            "search",
            "select",
            "choose",
            "pick",
            "move",
            "copy",
            "paste",
            "share",
            "download",
            "upload",
            "install",
            "setup",
            "configure",
            "manage",
            # Common tech infrastructure terms
            "application",
            "program",
            "software",
            "system",
            "platform",
            "website",
            "service",
            "server",
            "client",
            "database",
            "network",
            "interface",
            "feature",
            "function",
            "method",
            "process",
            "option",
            "setting",
            # Status/State terms
            "status",
            "state",
            "condition",
            "level",
            "stage",
            "phase",
            "step",
            "progress",
            "result",
            "output",
            "error",
            "warning",
            "success",
            "fail",
            # Common work/business terms
            "report",
            "review",
            "analysis",
            "strategy",
            "plan",
            "task",
            "meeting",
            "email",
            "call",
            "user",
            "customer",
            "client",
            "staff",
            "team",
            "group",
            "member",
            "employee",
            "manager",
            "department",
            # Common modifiers and connectors
            "dont",
            "cant",
            "wont",
            "its",
            "thats",
            "think",
            "know",
            "need",
            "want",
            "look",
            "come",
            "work",
            "make",
            "take",
            "use",
            "using",
            "used",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
        }

        # Generate variations
        additional_words = set()
        for word in word_stems:
            # Add original word
            additional_words.add(word)
            # Add simple plural
            additional_words.add(word + "s")
            # Add 'es' plural for words ending in s, sh, ch, x, z
            if word.endswith(("s", "sh", "ch", "x", "z")):
                additional_words.add(word + "es")
            # Add 'ies' plural for words ending in y
            elif word.endswith("y"):
                additional_words.add(word[:-1] + "ies")
            # Add 'ing' form
            additional_words.add(word + "ing")
            # Add 'ed' form
            additional_words.add(word + "ed")

        base_stop_words.update(additional_words)
        return base_stop_words

    def _initialize_technical_terms(self):
        """Initialize set of known technical terms to boost."""
        return {
            "api",
            "sdk",
            "ui",
            "framework",
            "library",
            "database",
            "server",
            "client",
            "network",
            "protocol",
            "algorithm",
            "kubernetes",
            "docker",
            "cloud",
            "git",
            "aws",
            "azure",
            "python",
            "javascript",
            "java",
            "golang",
            "rust",
            "react",
            "angular",
            "vue",
            "node",
            "express",
            "django",
            "flask",
        }

    def _is_meaningful_keyword(self, word, min_length=4, max_length=30):
        """Check if a word is likely to be a meaningful keyword."""
        if not min_length <= len(word) <= max_length:
            return False

        patterns_to_filter = [
            r"^[0-9]+$",  # Just numbers
            r"^(don\'?t|can\'?t|won\'?t|it\'?s|that\'?s|i\'?m|you\'?re|they\'?re)$",
            r"^(also|just|like|even|much|many|some|any|very|quite|rather|such|another|same)$",
            r"^(this|that|these|those|here|there|where|when|what|who|why|how)$",
            r"^(and|but|or|yet|for|nor|so|at|in|on|to|of|with|by)$",
        ]

        for pattern in patterns_to_filter:
            if re.match(pattern, word.lower()):
                return False

        return True

    def calculate_keyword_score(self, word, frequency, total_words):
        """Calculate a weighted score for each potential keyword."""
        freq_score = frequency / total_words

        technical_boost = 1.5 if word.lower() in self.technical_terms else 1.0
        specialty_boost = 1.3 if re.match(r"^[a-z]+[A-Z][a-zA-Z]*$", word) else 1.0
        length_boost = 1.0 + (0.1 * min(len(word) / 10, 1.0))

        # Network boost - favor terms we've seen before
        network_boost = 1.2 if word in self.existing_tags else 1.0

        return (
            freq_score
            * technical_boost
            * specialty_boost
            * length_boost
            * network_boost
        )

    def extract_key_phrases(self, text):
        """Extract meaningful multi-word phrases."""
        phrases = []
        patterns = [
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",  # Proper noun phrases
            r"[a-z]+(?:\s+[a-z]+){1,2}(?:\s+(?:API|SDK|UI|Framework|Library|System))",
            r"(?:machine|deep)\s+learning",
            r"(?:continuous|automated)\s+(?:integration|deployment|testing)",
            r"(?:artificial|business)\s+intelligence",
            r"(?:version|source)\s+control",
            r"(?:cloud|web)\s+(?:computing|service|platform)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            phrases.extend(m.group(0) for m in matches)

        return phrases

    def should_merge_keywords(self, word1, word2):
        """Determine if two keywords should be merged."""
        # Check for singular/plural forms
        if word1.endswith("s") and word1[:-1] == word2:
            return True
        if word2.endswith("s") and word2[:-1] == word1:
            return True

        # Check for common variations
        if word1.replace("-", "") == word2:
            return True
        if word1.replace("_", "") == word2:
            return True

        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
        if similarity > 0.8:  # 80% similar
            return True

        return False

    def clean_html(self, html_content):
        """Remove HTML tags and clean text."""
        if pd.isna(html_content):
            return ""

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text = soup.get_text(separator=" ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            print(f"Warning: Error cleaning HTML: {e}")
            return str(html_content)

    def extract_keywords(self, text, max_keywords=5):
        """Extract keywords with improved relevance filtering."""
        clean_text = self.clean_html(text)
        words = clean_text.lower().split()
        total_words = len(words)

        word_freq = Counter(words)
        key_phrases = self.extract_key_phrases(clean_text)

        candidates = []
        for word, freq in word_freq.items():
            if self._is_meaningful_keyword(word) and word not in self.stop_words:
                score = self.calculate_keyword_score(word, freq, total_words)
                candidates.append((word, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        final_keywords = []
        used_terms = set()

        # Add key phrases first
        for phrase in key_phrases[:max_keywords]:
            if len(final_keywords) >= max_keywords:
                break
            final_keywords.append(phrase)
            used_terms.add(phrase.lower())

        # Then add individual keywords
        for word, score in candidates:
            if len(final_keywords) >= max_keywords:
                break

            # Check if this word should be merged with existing keywords
            should_add = True
            for existing in final_keywords:
                if self.should_merge_keywords(word, existing):
                    should_add = False
                    break

            if should_add and word not in used_terms:
                final_keywords.append(word)
                used_terms.add(word)

                # Update network tracking
                self.existing_tags.add(word)
                self.tag_frequency[word] += 1

                # Update relationships with other keywords in this set
                for other_keyword in final_keywords:
                    if other_keyword != word:
                        self.tag_relationships[word].add(other_keyword)
                        self.tag_relationships[other_keyword].add(word)

        return final_keywords

    def generate_network_data(self):
        """Generate network analysis data."""
        nodes = []
        edges = []

        # Create nodes with frequency information
        for tag, freq in self.tag_frequency.items():
            nodes.append(
                {
                    "id": tag,
                    "frequency": freq,
                    "connections": len(self.tag_relationships[tag]),
                }
            )

        # Create edges from relationships
        seen_pairs = set()
        for tag1, related_tags in self.tag_relationships.items():
            for tag2 in related_tags:
                if (tag1, tag2) not in seen_pairs and (tag2, tag1) not in seen_pairs:
                    edges.append(
                        {
                            "source": tag1,
                            "target": tag2,
                            "weight": min(
                                self.tag_frequency[tag1], self.tag_frequency[tag2]
                            ),
                        }
                    )
                    seen_pairs.add((tag1, tag2))

        return {"nodes": nodes, "edges": edges}


def process_entries(csv_file):
    """Process entries and generate network data"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Initialize extractor
        extractor = NetworkKeywordExtractor()

        # Process unarchived entries
        unarchived_entries = df[~df["is_archived"]]

        results = []
        total_entries = len(unarchived_entries)

        print(f"Processing {total_entries} unarchived entries...")

        for count, (_, entry) in enumerate(unarchived_entries.iterrows(), 1):
            try:
                keywords = extractor.extract_keywords(entry["content"])
                if keywords:
                    results.append(
                        {
                            "entry_id": entry["id"],
                            "keywords": keywords,
                            "title": entry["title"],
                        }
                    )
                if count % 10 == 0:
                    print(f"Processed {count}/{total_entries} entries...")
            except Exception as e:
                print(f"Warning: Error processing entry {entry['id']}: {e}")
                continue

        print(f"Completed processing {count}/{total_entries} entries.")

        # Generate network data
        network_data = extractor.generate_network_data()

        return results, network_data

    except Exception as e:
        print(f"Error: Failed to process entries: {e}")
        return [], None


def generate_sql(results, existing_max_tag_id):
    """Generate SQL statements for tag insertion with network awareness"""
    sql_statements = []
    next_tag_id = existing_max_tag_id + 1
    tag_ids = {}  # Track tag IDs for reuse

    # First, generate all tag insertions
    sql_statements.append("-- First, insert all new tags\n")

    for entry in results:
        for keyword in entry["keywords"]:
            try:
                # Create slug from keyword
                slug = re.sub(r"[^\w\s-]", "", keyword.lower())
                slug = re.sub(r"[-\s]+", "-", slug).strip("-")

                if slug not in tag_ids:
                    # Escape single quotes in keyword
                    safe_keyword = keyword.replace("'", "''")

                    sql_statements.append(
                        f"""
INSERT INTO public.wallabag_tag (id, label, slug)
SELECT {next_tag_id}, '{safe_keyword}', '{slug}'
WHERE NOT EXISTS (
    SELECT 1 FROM public.wallabag_tag WHERE slug = '{slug}'
);
"""
                    )
                    tag_ids[slug] = next_tag_id
                    next_tag_id += 1

            except Exception as e:
                print(f"Warning: Error generating SQL for keyword '{keyword}': {e}")
                continue

    # Then, generate all entry-tag relationships
    sql_statements.append("\n-- Then, create entry-tag relationships\n")

    for entry in results:
        entry_id = entry["entry_id"]
        for keyword in entry["keywords"]:
            try:
                slug = re.sub(r"[^\w\s-]", "", keyword.lower())
                slug = re.sub(r"[-\s]+", "-", slug).strip("-")

                sql_statements.append(
                    f"""
WITH tag_id_lookup AS (
    SELECT id FROM public.wallabag_tag WHERE slug = '{slug}'
)
INSERT INTO public.wallabag_entry_tag (entry_id, tag_id)
SELECT {entry_id}, tag_id_lookup.id
FROM tag_id_lookup
WHERE NOT EXISTS (
    SELECT 1 FROM public.wallabag_entry_tag
    WHERE entry_id = {entry_id} 
    AND tag_id = tag_id_lookup.id
);
"""
                )

            except Exception as e:
                print(f"Warning: Error generating SQL for entry-tag relationship: {e}")
                continue

    return "\n".join(sql_statements)


if __name__ == "__main__":
    # Configuration
    CSV_FILE = "YOUR_CSV_HERE.csv"
    EXISTING_MAX_TAG_ID = 0

    # Process entries and generate network data
    print("Starting entry processing...")
    results, network_data = process_entries(CSV_FILE)

    if results:
        # Generate SQL
        print(f"Generating SQL for {len(results)} entries...")
        sql_statements = generate_sql(results, EXISTING_MAX_TAG_ID)

        # Save SQL statements
        try:
            with open("tag_insertions.sql", "w") as f:
                f.write(sql_statements)
            print("SQL statements written to tag_insertions.sql")
        except Exception as e:
            print(f"Error writing SQL: {e}")

        # Save network data
        try:
            with open("tag_network.json", "w") as f:
                json.dump(network_data, f, indent=2)
            print("Network data written to tag_network.json")
        except Exception as e:
            print(f"Error writing network data: {e}")
    else:
        print("No results generated. Check the error messages above.")
