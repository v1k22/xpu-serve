"""Reads a GGUF header (metadata and tensor table) from the first bytes of a file. Read-only analysis helper."""
import struct, sys, json, collections

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0"}
# bytes per block, weights per block
BLOCK = {"F32": (4, 1), "F16": (2, 1), "BF16": (2, 1), "Q4_0": (18, 32), "Q4_1": (20, 32), "Q5_0": (22, 32), "Q5_1": (24, 32),
         "Q8_0": (34, 32), "Q4_K": (144, 256), "Q5_K": (176, 256), "Q6_K": (210, 256), "Q3_K": (110, 256), "Q2_K": (84, 256),
         "Q8_K": (292, 256), "I8": (1, 1), "I16": (2, 1), "I32": (4, 1), "I64": (8, 1), "F64": (8, 1)}

class R:
    def __init__(self, b): self.b, self.p = b, 0
    def u(self, fmt):
        n = struct.calcsize(fmt); v = struct.unpack_from("<" + fmt, self.b, self.p)[0]; self.p += n; return v
    def s(self):
        n = self.u("Q"); v = self.b[self.p:self.p + n]; self.p += n; return v.decode("utf-8", "replace")
    def val(self, t):
        if t == 8: return self.s()
        if t == 9:
            et = self.u("I"); n = self.u("Q")
            return [self.val(et) for _ in range(n)]
        return self.u({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t])

def parse(path):
    r = R(open(path, "rb").read())
    assert r.b[:4] == b"GGUF"; r.p = 4
    version = r.u("I"); n_tensors = r.u("Q"); n_kv = r.u("Q")
    kv = collections.OrderedDict()
    for _ in range(n_kv):
        k = r.s(); t = r.u("I"); kv[k] = r.val(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.s(); nd = r.u("I"); dims = [r.u("Q") for _ in range(nd)]; t = r.u("I"); off = r.u("Q")
        tensors.append({"name": name, "dims": dims, "type": TYPES.get(t, str(t)), "offset": off})
    align = kv.get("general.alignment", 32)
    data_start = (r.p + align - 1) // align * align
    return version, kv, tensors, data_start

if __name__ == "__main__":
    version, kv, tensors, data_start = parse(sys.argv[1])
    print("version", version, "| kv", len(kv), "| tensors", len(tensors), "| data starts at", data_start)
    for k, v in kv.items():
        if isinstance(v, list): print(f"  {k}: list[{len(v)}] {str(v[:6])[:110]}")
        else: print(f"  {k}: {str(v)[:150]!r}" if isinstance(v, str) else f"  {k}: {v}")
    hist = collections.Counter(); byt = collections.Counter()
    for t in tensors:
        n = 1
        for d in t["dims"]: n *= d
        bb, wb = BLOCK.get(t["type"], (0, 1))
        hist[t["type"]] += 1; byt[t["type"]] += n // wb * bb
    print("types:", dict(hist)); print("bytes by type (GB):", {k: round(v / 1e9, 2) for k, v in byt.items()})
    json.dump({"kv": {k: (v if not isinstance(v, list) or len(v) < 64 else f"list[{len(v)}]") for k, v in kv.items()},
               "tensors": tensors, "data_start": data_start}, open(sys.argv[1] + ".json", "w"))
