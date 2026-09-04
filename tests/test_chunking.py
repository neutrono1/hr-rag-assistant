import unittest
from app.chunking import chunk_markdown


SAMPLE = """# Leave Policy

## 2. Leave types

### 2.1 Casual leave (CL)

Employees receive **12 casual leave days** per calendar year.

## 4. Carry-forward

### 4.1 Casual leave carry-forward

A maximum of **8 days** of unused casual leave may be carried forward.

| Band | Annual LTA limit |
| --- | --- |
| Band A | 25000 |
| Band B | 50000 |
"""


class TestChunking(unittest.TestCase):
    def test_section_paths_are_tracked(self):
        chunks = chunk_markdown("Leave Policy", SAMPLE)
        sick_chunk = [c for c in chunks if "12 casual leave days" in c.text]
        self.assertEqual(len(sick_chunk), 1)
        self.assertIn("2.1 Casual leave (CL)", sick_chunk[0].section_path)
        self.assertIn("2. Leave types", sick_chunk[0].section_path)

    def test_table_rows_are_split(self):
        chunks = chunk_markdown("Leave Policy", SAMPLE)
        row_chunks = [c for c in chunks if c.chunk_type == "table_row"]
        self.assertEqual(len(row_chunks), 2)
        self.assertIn("Band A", row_chunks[0].text)
        self.assertIn("25000", row_chunks[0].text)

    def test_full_table_chunk_exists(self):
        chunks = chunk_markdown("Leave Policy", SAMPLE)
        full = [c for c in chunks if c.chunk_type == "table_full"]
        self.assertEqual(len(full), 1)
        self.assertIn("Band A", full[0].text)
        self.assertIn("Band B", full[0].text)

    def test_no_chunk_crosses_a_heading(self):
        chunks = chunk_markdown("Leave Policy", SAMPLE)
        for c in chunks:
            self.assertNotIn("##", c.text)


if __name__ == "__main__":
    unittest.main()
