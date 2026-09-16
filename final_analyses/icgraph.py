"""Ontology DAG with information content, for Lin and Wu-Palmer similarity.

Parses an OBO file (is_a, optionally part_of) or an OWL file (rdfs:subClassOf
with a named parent). IC(c) = -log2((|descendants(c)| + 1) / (N + 1)),
Lin(a, b) = 2 IC(MICA) / (IC(a) + IC(b)). Shared by the staircase (Figure 2),
PRIDE-context (Figure 4) and matcher analyses.
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict


class IcGraph:
    def __init__(self, path, relations=("is_a", "part_of")):
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.terms = set()
        self._parse(path, relations)
        self.N = len(self.terms)
        sys.setrecursionlimit(20000)
        self._anc, self._desc, self._depth = {}, {}, {}

    # ---- parsing -----------------------------------------------------------
    def _parse(self, path, relations):
        if str(path).endswith(".owl"):
            return self._parse_owl(path)
        cur, obsolete = None, False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "[Term]":
                    cur, obsolete = None, False
                elif line.startswith("[") and line.endswith("]"):
                    cur = None
                elif cur is None and line.startswith("id: "):
                    cur = line[4:].strip()
                elif cur:
                    if line.startswith("is_obsolete: true"):
                        self.terms.discard(cur); obsolete = True
                    elif obsolete:
                        continue
                    elif line.startswith("is_a: "):
                        self.terms.add(cur); self._edge(cur, line[6:].split("!")[0].split("{")[0].strip())
                    elif line.startswith("relationship: "):
                        parts = line[14:].split("!")[0].split("{")[0].split()
                        if len(parts) >= 2 and parts[0] in relations:
                            self.terms.add(cur); self._edge(cur, parts[1])
                    elif line.startswith("name: "):
                        self.terms.add(cur)

    def _parse_owl(self, path):
        cls_re = re.compile(r'<owl:Class rdf:about="([^"]+)"')
        sub_re = re.compile(r'<rdfs:subClassOf rdf:resource="([^"]+)"')
        dep_re = re.compile(r'<owl:deprecated[^>]*>true<')

        def curie(uri):
            tail = uri.rsplit("/", 1)[-1]
            return tail.replace("_", ":", 1) if "_" in tail else tail
        cur = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = cls_re.search(line)
                if m:
                    cur = curie(m.group(1)); self.terms.add(cur); continue
                if cur is None:
                    continue
                if "</owl:Class>" in line:
                    cur = None
                elif dep_re.search(line):
                    self.terms.discard(cur)
                else:
                    m = sub_re.search(line)
                    if m:
                        self._edge(cur, curie(m.group(1)))

    def _edge(self, child, parent):
        if child != parent:
            self.parents[child].add(parent); self.children[parent].add(child)

    # ---- graph queries -----------------------------------------------------
    def _closure(self, t, rel, cache):
        if t not in cache:
            out, stack = set(), [t]
            while stack:
                n = stack.pop()
                for p in rel.get(n, ()):
                    if p not in out:
                        out.add(p); stack.append(p)
            cache[t] = out
        return cache[t]

    def ancestors(self, t):
        return self._closure(t, self.parents, self._anc)

    def descendants(self, t):
        return self._closure(t, self.children, self._desc)

    def ic(self, t):
        return -math.log2((len(self.descendants(t)) + 1) / (self.N + 1))

    def depth(self, t):
        """Shortest number of edges up to a root."""
        if t not in self._depth:
            if not self.parents.get(t):
                self._depth[t] = 0
            else:
                seen, frontier, d, found = {t}, [t], 0, None
                while frontier and found is None:
                    d += 1; nxt = []
                    for n in frontier:
                        for p in self.parents.get(n, ()):
                            if not self.parents.get(p):
                                found = d; break
                            if p not in seen:
                                seen.add(p); nxt.append(p)
                        if found is not None:
                            break
                    frontier = nxt
                self._depth[t] = found if found is not None else d
        return self._depth[t]

    def mica(self, a, b):
        common = (self.ancestors(a) | {a}) & (self.ancestors(b) | {b})
        return max(common, key=self.ic) if common else None

    def lin(self, a, b):
        if a == b:
            return 1.0
        m = self.mica(a, b)
        if m is None:
            return 0.0
        denom = self.ic(a) + self.ic(b)
        return 2 * self.ic(m) / denom if denom > 0 else 0.0

    def wu_palmer(self, a, b):
        if a == b:
            return 1.0
        m = self.mica(a, b)
        if m is None:
            return 0.0
        da, db = self.depth(a), self.depth(b)
        return min(1.0, 2 * self.depth(m) / (da + db)) if (da + db) > 0 else 0.0
