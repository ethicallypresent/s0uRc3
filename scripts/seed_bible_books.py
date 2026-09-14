#!/usr/bin/env python3
"""Seed a one-line index of all 66 books of the Bible into long-term memory.

Same mechanism as the other seed_bible_*.py scripts. Meant as the base
layer the other seeds (seed_bible_fundamentals.py, seed_bible_john.py,
seed_bible_tools.py) sit on top of — each entry leads with the book's name
so future recall (chat_agent.py's _recall_block, which re-ranks by keyword
overlap on top of embedding score) has a strong, exact match to anchor on
for "what's in the book of X"-style questions, not just an embedding
guess.

Usage:
  python scripts/seed_bible_books.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.memory import Memory
from core.paths import AgentPaths

OLD_TESTAMENT: list[str] = [
    "Genesis: the beginning — creation, the Fall, the flood, and the patriarchs Abraham, Isaac, Jacob, and Joseph.",
    "Exodus: Israel's slavery in Egypt, the plagues, the Passover, and the exodus led by Moses, ending at Mount Sinai.",
    "Leviticus: the priestly law — sacrifices, purity, and holiness codes for Israel.",
    "Numbers: Israel's census and forty years wandering the wilderness after leaving Sinai.",
    "Deuteronomy: Moses' farewell speeches restating the law to Israel before entering the Promised Land.",
    "Joshua: Israel's conquest and settlement of the Promised Land under Joshua's leadership.",
    "Judges: the cycle of Israel's disobedience, oppression, and deliverance through judges like Deborah, Gideon, and Samson.",
    "Ruth: a Moabite widow's loyalty to her mother-in-law Naomi, and her marriage to Boaz, an ancestor of King David.",
    "1 Samuel: Israel's transition to monarchy — Samuel, King Saul's rise and fall, and David's anointing.",
    "2 Samuel: King David's reign, his triumphs, his sin with Bathsheba, and the troubles that follow.",
    "1 Kings: Solomon's reign and the temple's construction, then the kingdom splits into Israel and Judah.",
    "2 Kings: the decline and fall of Israel and Judah, ending in exile to Assyria and Babylon.",
    "1 Chronicles: a priestly retelling of Israel's history from Adam through King David's reign.",
    "2 Chronicles: a priestly retelling of Solomon's reign through the fall of Judah and the Babylonian exile.",
    "Ezra: the Jewish return from Babylonian exile and the rebuilding of the temple in Jerusalem.",
    "Nehemiah: Nehemiah's leadership rebuilding Jerusalem's walls despite opposition.",
    "Esther: a Jewish queen in Persia who risks her life to save her people from destruction.",
    "Job: a righteous man's suffering and his dialogue with friends and God over the meaning of undeserved pain.",
    "Psalms: 150 poems and songs of worship, lament, praise, and prayer, many attributed to David.",
    "Proverbs: short sayings of practical wisdom, largely attributed to Solomon.",
    "Ecclesiastes: a meditation, attributed to Solomon, on the meaninglessness of life apart from God.",
    "Song of Songs (Song of Solomon): a love poem celebrating romantic and marital love.",
    "Isaiah: a major prophet's warnings of judgment and promises of a coming Messiah and restoration.",
    "Jeremiah: a major prophet's warnings to Judah before the Babylonian exile, and his own suffering for delivering them.",
    "Lamentations: five poems of mourning over the destruction of Jerusalem.",
    "Ezekiel: a major prophet's visions during the Babylonian exile, of judgment and Israel's eventual restoration.",
    "Daniel: a Jewish exile in Babylon whose faithfulness under pressure and apocalyptic visions are recorded.",
    "Hosea: a minor prophet whose troubled marriage becomes a picture of Israel's unfaithfulness to God.",
    "Joel: a minor prophet's call to repentance after a locust plague, and a promise of God's Spirit poured out.",
    "Amos: a minor prophet's warnings against social injustice and empty religious ritual in Israel.",
    "Obadiah: the shortest Old Testament book, a minor prophet's judgment against Edom for its treatment of Israel.",
    "Jonah: a reluctant minor prophet sent to warn Nineveh, swallowed by a great fish after fleeing his call.",
    "Micah: a minor prophet's warnings of judgment and a promise that a ruler would come from Bethlehem.",
    "Nahum: a minor prophet's prophecy of judgment against Nineveh.",
    "Habakkuk: a minor prophet's dialogue with God questioning why the wicked prosper.",
    "Zephaniah: a minor prophet's warning of the coming Day of the Lord's judgment.",
    "Haggai: a minor prophet urging the returned exiles to finish rebuilding the temple.",
    "Zechariah: a minor prophet's visions encouraging the rebuilding of the temple and pointing to a future king.",
    "Malachi: the last Old Testament book, a minor prophet rebuking Israel's half-hearted worship.",
]

NEW_TESTAMENT: list[str] = [
    "Matthew: a Gospel presenting Jesus as the promised Jewish Messiah, with the Sermon on the Mount.",
    "Mark: the shortest Gospel, presenting Jesus as a man of action, moving quickly from miracle to miracle.",
    "Luke: a Gospel written for a broader audience, emphasizing Jesus' compassion for the poor, outcast, and Gentiles.",
    "John: a Gospel presenting Jesus as the divine Word made flesh, built around seven signs and seven 'I am' statements.",
    "Acts: the early church's growth after Jesus' ascension, the coming of the Holy Spirit, and Paul's missionary journeys.",
    "Romans: Paul's letter laying out the theology of sin, grace, and salvation by faith.",
    "1 Corinthians: Paul's letter correcting division, immorality, and confusion in the Corinthian church.",
    "2 Corinthians: Paul's letter defending his ministry and apostleship to the Corinthian church.",
    "Galatians: Paul's letter insisting salvation is by faith, not by keeping the Jewish law.",
    "Ephesians: Paul's letter on the church as the body of Christ, unity, and spiritual armor.",
    "Philippians: Paul's letter of joy and encouragement written from prison.",
    "Colossians: Paul's letter emphasizing the supremacy of Christ over all things.",
    "1 Thessalonians: Paul's letter encouraging a young church and addressing Christ's return.",
    "2 Thessalonians: Paul's follow-up letter correcting confusion about the timing of Christ's return.",
    "1 Timothy: Paul's letter to a young pastor on church leadership and order.",
    "2 Timothy: Paul's final letter, written from prison, urging Timothy to remain faithful.",
    "Titus: Paul's letter to Titus on appointing elders and sound teaching in the churches of Crete.",
    "Philemon: Paul's short personal letter appealing for forgiveness for a runaway slave, Onesimus.",
    "Hebrews: a letter arguing Christ is superior to the old covenant's priests, sacrifices, and law.",
    "James: a practical letter on living out genuine faith through works, wisdom, and speech.",
    "1 Peter: Peter's letter encouraging Christians to endure suffering and persecution with hope.",
    "2 Peter: Peter's letter warning against false teachers and affirming Christ's certain return.",
    "1 John: John's letter on assurance of salvation, love for one another, and testing false teaching.",
    "2 John: John's short letter warning against supporting false teachers.",
    "3 John: John's short personal letter commending hospitality to traveling teachers.",
    "Jude: a short letter warning against false teachers who had infiltrated the church.",
    "Revelation: John's apocalyptic visions of the end times, Christ's return, and the new heaven and earth.",
]


def main() -> None:
    paths = AgentPaths.discover()
    memory = Memory(paths.db)
    saved = 0
    skipped = 0
    try:
        for text in OLD_TESTAMENT + NEW_TESTAMENT:
            eid = memory.save(text, kind="fact", verified=True)
            if eid:
                saved += 1
            else:
                skipped += 1
    finally:
        memory.close()
    print(f"Seeded {saved} book-index memory entries ({skipped} skipped as near-duplicates).")


if __name__ == "__main__":
    main()
