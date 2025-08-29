import unittest

from main import slugify


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_multiple_spaces(self):
        self.assertEqual(slugify("Hello   World"), "hello---world")

    def test_leading_trailing_spaces(self):
        self.assertEqual(slugify("  Hello World  "), "--hello-world--")

    def test_unicode_letters(self):
        self.assertEqual(slugify("Café au lait"), "café-au-lait")

    def test_punctuation(self):
        self.assertEqual(slugify("Hello, World!"), "hello,-world!")

    def test_empty_string(self):
        self.assertEqual(slugify(""), "")

    def test_no_spaces(self):
        self.assertEqual(slugify("Already-slugified"), "already-slugified")

    # Additional edge cases
    def test_tabs_preserved(self):
        # Tabs are not ASCII spaces; they should not be replaced
        self.assertEqual(slugify("Hello\tWorld"), "hello\tworld")

    def test_non_breaking_space_preserved(self):
        # NBSP (\u00A0) should not be replaced because we only replace ASCII space
        s = "Hello\u00A0World"
        self.assertEqual(slugify(s), "hello\u00a0world")

    def test_mixed_whitespace(self):
        s = "Hello \t World"
        self.assertEqual(slugify(s), "hello-\t-world")

    def test_only_spaces(self):
        self.assertEqual(slugify("     "), "-----")

    def test_long_input(self):
        s = ("A" * 1000) + " " + ("B" * 1000)
        expected = ("a" * 1000) + "-" + ("b" * 1000)
        self.assertEqual(slugify(s), expected)


if __name__ == "__main__":
    unittest.main()


