#include "convert/calibrate.h"

namespace xs {

// Written for xpu-serve (MIT, like the rest of the repository). Varied on purpose: activation statistics should cover
// the kinds of text a model sees, not one domain.
const std::string& builtin_calibration_text() {
    static const std::string text = R"xs(
The river town woke slowly. Fishermen untied their boats before the sun cleared the hills, and the baker on the corner
pulled the first trays of bread from an oven that had been warming since midnight. By seven the market square was
loud with traders arguing about the price of onions, a dog chasing pigeons, and a radio somewhere playing an old song
about the sea. Mira walked through it all with a notebook in her pocket, counting the stalls, because the council had
asked her to measure how many people the square could hold on festival day.

Photosynthesis converts light energy into chemical energy. In the chloroplasts of plant cells, chlorophyll absorbs
mostly red and blue light, and the energy drives the splitting of water into oxygen, protons and electrons. The
electrons pass along a chain of proteins in the thylakoid membrane, pumping protons that power ATP synthase. In the
Calvin cycle the enzyme RuBisCO fixes carbon dioxide onto a five-carbon sugar, and with ATP and NADPH the cell builds
glucose. Roughly six molecules of carbon dioxide and six of water yield one of glucose and six of oxygen.

Newton's second law states that the net force on a body equals its mass times its acceleration, F = m a. A 2 kg cart
pushed with a net force of 10 N accelerates at 5 m/s^2. If it starts from rest, after 3 seconds its speed is 15 m/s and
it has travelled 22.5 m, since distance equals one half of a times t squared. Kinetic energy is one half m v^2, so the
cart then carries 225 J, which matches the work done: force times distance, 10 N times 22.5 m.

Question: A train leaves at 9:40 and arrives at 13:05. How long is the journey?
Answer: From 9:40 to 12:40 is 3 hours, and from 12:40 to 13:05 is 25 minutes, so the journey takes 3 hours 25 minutes.

Question: Solve 3x + 7 = 25.
Answer: Subtract 7 from both sides to get 3x = 18, then divide by 3: x = 6. Check: 3 * 6 + 7 = 25.

Question: What is the derivative of f(x) = x^3 - 4x + 2?
Answer: f'(x) = 3x^2 - 4. At x = 2 the slope is 8, so the tangent line there is y = 8x - 14.

The printing press, developed in Europe in the fifteenth century, changed how ideas travelled. Before it, books were
copied by hand and were rare and costly. Movable metal type allowed a workshop to produce hundreds of identical pages
in a day. Within fifty years printed books numbered in the millions, literacy spread among merchants and craftsmen, and
scholars could compare the same text in different cities. Historians still argue about how much the press caused the
religious and scientific changes that followed, and how much it simply sped up changes already under way.

def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

#include <vector>
#include <numeric>

// Returns the running sums of v: out[i] = v[0] + ... + v[i].
std::vector<long> prefix_sums(const std::vector<long>& v) {
    std::vector<long> out(v.size());
    std::partial_sum(v.begin(), v.end(), out.begin());
    return out;
}

SELECT customer_id, COUNT(*) AS orders, SUM(total) AS spent
FROM orders
WHERE created_at >= '2025-01-01'
GROUP BY customer_id
HAVING SUM(total) > 1000
ORDER BY spent DESC;

User: My laptop gets very hot when I watch videos. Is that normal?
Assistant: Some warmth is normal, because video decoding keeps the processor busy. If it gets too hot to touch, check
that the air vents are not blocked, clean any dust from the fan, and make sure the browser uses hardware video
decoding. Keeping the laptop on a hard surface instead of a blanket also helps air flow.

User: Can you summarize the main causes of inflation in two sentences?
Assistant: Inflation rises when demand for goods and services grows faster than the economy can produce them, or when
costs such as energy and wages push prices up. Expectations matter too: if people expect prices to keep rising, they
spend and set prices in ways that make it happen.

The human heart has four chambers. Blood low in oxygen returns through the veins to the right atrium, passes to the
right ventricle and is pumped to the lungs, where it picks up oxygen and releases carbon dioxide. It comes back to the
left atrium, and the thick-walled left ventricle pumps it into the aorta and on to the body. Valves between the
chambers keep the flow moving in one direction, and their closing makes the familiar two-part sound of a heartbeat.

In computer science, a hash table stores key-value pairs so that lookups take constant time on average. A hash
function maps each key to a slot in an array. When two keys land in the same slot, the table resolves the collision,
either by keeping a short list in each slot or by probing for the next free one. As the table fills, collisions
become more frequent, so implementations grow the array and rehash every entry once the load factor passes a limit.

Een klein dorp aan de kust leefde eeuwenlang van de visserij. Tegenwoordig komen er vooral toeristen, die in de zomer
over het strand wandelen en 's avonds verse vis eten in de haven.

La biblioteca abre a las nueve de la mañana. Los estudiantes llegan temprano para conseguir una mesa junto a la
ventana, donde la luz es mejor para leer, y se quedan hasta que la bibliotecaria anuncia el cierre.

Le climat de la région est doux en hiver et chaud en été. Les vignes couvrent les collines, et chaque automne les
villages fêtent la fin des vendanges avec de la musique et de longues tables dressées dans les rues.

भारत में मानसून जून के आसपास आता है। किसान बारिश का इंतज़ार करते हैं, क्योंकि धान और कई दूसरी फ़सलों की बुवाई
इसी समय होती है। अच्छी बारिश से गाँवों में ख़ुशी लौट आती है।

春天到了，公园里的樱花都开了。周末很多人带着家人来散步、拍照，孩子们在草地上放风筝，老人们坐在长椅上聊天。

東京の朝の電車はとても混んでいます。多くの人がスマートフォンでニュースを読みながら、静かに目的地に向かいます。

Die Stadt plant eine neue Brücke über den Fluss. Sie soll Fußgängern und Radfahrern vorbehalten sein und die beiden
Ufer in nur wenigen Minuten verbinden.

A good experiment controls everything except the variable under study. To test whether a fertilizer helps tomato
plants, a gardener grows two groups from the same seeds, in the same soil, with the same water and light, and feeds
only one group. Measuring the yield of many plants, not just one of each, guards against the result being chance.
If the treated group produces more fruit across several seasons, the evidence for the fertilizer grows stronger.

The Silk Road was not a single road but a web of routes linking China, Central Asia, Persia and the Mediterranean.
Silk, spices, paper and porcelain moved west; horses, glass, wool and silver moved east. Just as important were the
ideas that travelled with the goods: religions, languages, mathematics and techniques for making paper spread along
the same paths, carried by merchants, monks and soldiers over many centuries.

Table: monthly rainfall in millimetres
| Month | City A | City B |
|-------|--------|--------|
| Jan   | 78     | 12     |
| Apr   | 54     | 30     |
| Jul   | 20     | 210    |
| Oct   | 66     | 95     |
City A has its wettest month in January, while City B gets most of its rain in July, which suggests a monsoon climate.

{"id": 1042, "name": "sensor-7", "location": {"lat": 47.37, "lon": 8.54}, "readings": [21.4, 21.9, 22.3],
 "status": "ok", "last_seen": "2026-03-14T08:30:00Z"}

Proof that the square root of 2 is irrational: suppose it equals p/q in lowest terms. Then p^2 = 2 q^2, so p^2 is
even, and therefore p is even; write p = 2k. Substituting gives 4 k^2 = 2 q^2, so q^2 = 2 k^2 and q is even as well.
Both p and q being even contradicts the assumption that the fraction was in lowest terms, so no such fraction exists.

To make a simple tomato sauce, warm olive oil in a pan, add a chopped onion and cook it gently until soft. Add two
cloves of garlic for one minute, then a tin of crushed tomatoes, a pinch of salt and a few basil leaves. Let it
simmer for twenty minutes, stirring now and then, until it thickens. Taste it and adjust the salt before serving.
)xs";
    return text;
}

}  // namespace xs
