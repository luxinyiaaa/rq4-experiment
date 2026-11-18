import unittest

from scripts.build_kb_index import _make_catalogues, record_name


class CatalogueIndexTests(unittest.TestCase):
    def test_etc_archive_keeps_numeric_site_suffix_and_has_five_members(self):
        sites = ["Lonzee", "Monte Bondone", "Renon", "San Rossore 2", "Svartberget"]
        pages = []
        for site in sites:
            title, truncated, _ = record_name(
                f"# ETC L2 ARCHIVE from {site}\nStation\n{site}\nStart time (UTC)\n2024-01-01\nStop time (UTC)\n2024-12-31"
            )
            self.assertFalse(truncated)
            pages.append({"url": f"https://meta.icos-cp.eu/objects/{site.replace(' ', '')}0123456789",
                          "url_prefix": "meta.icos-cp.eu/objects", "record_name": title,
                          "name_truncated": truncated, "page_type": "dataset_record",
                          "fields": {"station": site, "start time": "2024-01-01", "stop time": "2024-12-31"}})
        catalogues = _make_catalogues(pages)
        etc = [row for row in catalogues if row["title_template"].startswith("etc l2 archive")]
        self.assertEqual(len(etc), 1)
        self.assertEqual(len(etc[0]["members"]), 5)
        self.assertIn("ETC L2 ARCHIVE from San Rossore 2", [m["title"] for m in etc[0]["members"]])


if __name__ == "__main__":
    unittest.main()
