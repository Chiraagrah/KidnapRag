import json, datetime
P = r"D:\DKD\NJIT Courses\RAG Security\KidnapRag\dataset_builder\results_sample\chunk_00.json"
d = json.load(open(P, encoding="utf-8"))

def mk(items, snips, errs):
    out = []
    for i, (t, u) in enumerate(items):
        r = {"title": t, "url": u, "snippet": ""}
        if i in snips:
            r["snippet"] = snips[i]
        else:
            r["fetch_error"] = errs.get(i, "not_fetched_beyond_top_3")
        out.append(r)
    return out

S = "https://www.sensacine.com.mx/actores/actor-99042/biografia/"
S2 = "https://www.sensacine.com/actores/actor-99042/biografia/"
E = "https://www.entityfacts.com/en/person/scott-derrickson"
sc_mx = 'Nacionalidad: Americano. "nació el 16 de julio de 1966 en Denver, Colorado, Estados Unidos." Quick facts also list Nacimiento 1977 (Denver, Colorado - Estados Unidos).'
sc_es = 'Nacionalidad: Americano. Breadcrumb: "Guionista americano." Activities: Guionista, Director, Productor delegado. Biography lists birth date 18 de marzo de 1977; he lives in Glendale, California.'
http403 = "HTTP 403 Forbidden"

def sd(*urls_titles):
    return urls_titles

derr_hop = [("Scott Derrickson", S), ("Scott Derrickson", S2), ("scott derrickson", E),
 ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
 ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
 ("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
 ("Derrickson", "https://en.wikipedia.org/wiki/Derrickson"),
 ("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
 ("Scott Derrickson", "https://en.wikipedia.org/wiki/Scott_Derrickson"),
 ("m.imdb.com", "https://m.imdb.com/es/name/nm0220600/awards")]
derr_para = [("Scott Derrickson", S2), ("Scott Derrickson", S), ("scott derrickson", E),
 ("Scott Derrickson", "https://moviemanner.com/people/scott-derrickson"),
 ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
 ("m.imdb.com", "https://m.imdb.com/name/nm0220600"),
 ("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
 ("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
 ("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
 ("Scott Derrickson", "https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Scott_Derrickson.html")]
ipfs_skip = {9: "skipped_low_value_domain"}

enc = "https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr"
kidz = "https://wiki.kidzsearch.com/wiki/Ed_Wood"
tmdb = "https://themoviedb.org/person/44763-edward-d-wood-jr/translations"
enc_s = "Nationality: American. Born: Edward Davis Wood, Poughkeepsie, NY, 10 October 1924. (International Dictionary of Films and Filmmakers entry)"
tmdb_s = 'English biography (from Wikipedia): Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978), better known as Ed Wood, "was an American screenwriter, director, producer, actor, author, and editor."'
wood_hop = [("wood edward d jr", enc), ("Ed Wood", kidz), ("English (en-US)", tmdb),
 ("By Bill Kleppel", "https://poklib.org/ed-wood-b-movie-cult-icon-poughkeepsiean/"),
 ("Edward Wood", "https://en.wikipedia.org/wiki/Edward_Wood"),
 ("www.imdb.com", "https://www.imdb.com:443/name/nm0000248/mediaviewer/rm1486738176"),
 ("Ed Wood", "https://en.wikipedia.org/wiki/Ed_Wood"),
 ("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste", "https://www.scifi-universe.com/fiche_personne.asp?per_id=954"),
 ("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste", "https://www.scifi-universe.com/personnalites/954/edward-d-wood-jr"),
 ("By Bill Kleppel", "https://poklib.org/?p=38792")]
wood_para = [("wood edward d jr", enc), ("Ed Wood", kidz), ("English (en-US)", tmdb),
 ("data.cinemabelgica.be", "https://data.cinemabelgica.be/person/9855"),
 ("Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker,", "https://www.acmi.net.au/creators/27635"),
 ("www.imdb.com", "https://www.imdb.com:443/title/tt0109707/fullcredits/cast"),
 ("Ed Wood", "https://discordia.fandom.com/wiki/Ed_Wood"),
 ("g.co", "https://g.co/kg/m/02vt5"),
 ("14 campy facts about ed wood", "https://www.mentalfloss.com/entertainment/movies/14-campy-facts-about-ed-wood"),
 ("secure.paperbackswap.com", "https://secure.paperbackswap.com/Ed-Wood/author/?t=al")]

cmp_hop = [("Scott Derrickson", S), ("Scott Derrickson", S2), ("scott derrickson", E),
 ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
 ("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
 ("Scott Derrickson", "https://Www.Wikipedia.org/wiki/Bermuda_(film)"),
 ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
 ("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
 ("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
 ("movies.oktaycolakoglu.com", "https://movies.oktaycolakoglu.com/people/55499")]
cmp_para = [("C4 p", "https://biola.atlassian.net/wiki/x/C4-p"),
 ("English (en-US)", tmdb), ("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
 ("Scott Derrickson", S2), ("Scott Derrickson", "https://biola.atlassian.net/wiki/x/34Lp"),
 ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
 ("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
 ("Scott Derrickson", "https://m.the-numbers.com/people/directors/0SCDC.php"),
 ("m.imdb.com", "https://m.imdb.com/name/nm0220600"),
 ("Scott Derrickson", "https://streamin.co/ee/person/scott-derrickson-55499")]

specs = [
 ("hop", 0, "What nationality is Scott Derrickson?", derr_hop, {0: sc_mx, 1: sc_es}, {2: http403}),
 ("hop_paraphrase", 0, "Scott Derrickson nationality director", derr_para, {0: sc_es, 1: sc_mx + " Known for directing Doctor Strange (2016)."}, {2: http403}),
 ("hop", 1, "What nationality was Ed Wood?", wood_hop, {0: enc_s, 2: tmdb_s}, {1: http403}),
 ("hop_paraphrase", 1, "Ed Wood director nationality", wood_para, {0: enc_s, 2: tmdb_s}, {1: http403}),
 ("hop", 2, "Scott Derrickson Ed Wood same nationality comparison", cmp_hop, {0: sc_mx + " Page does not mention Ed Wood.", 1: sc_es + " Page does not mention Ed Wood."}, {2: http403}),
 ("hop_paraphrase", 2, "compare nationalities Scott Derrickson Ed Wood", cmp_para, {1: tmdb_s}, {0: "HTTP 202 empty body", 2: "HTTP 404 Not Found"}),
]
have = {(e["qid"], e["search_type"], e["hop_idx"]) for e in d}
for st, hi, q, items, sn, er in specs:
    if ("sample_3hop_c", st, hi) in have:
        continue
    d.append({"qid": "sample_3hop_c", "search_type": st, "hop_idx": hi, "query_text": q,
              "results": mk(items, sn, er),
              "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(d))
