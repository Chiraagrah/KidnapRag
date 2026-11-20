import json, datetime
P = r"D:\DKD\NJIT Courses\RAG Security\KidnapRag\dataset_builder\results_sample\chunk_00.json"
data = json.load(open(P, encoding="utf-8"))

SN = {
 "https://www.sensacine.com.mx/actores/actor-99042/biografia/": 'Quick-facts label "Nacionalidad" with value "Americano". Biography: "nació el 16 de julio de 1966 en Denver, Colorado, Estados Unidos."',
 "https://www.sensacine.com/actores/actor-99042/biografia/": 'Quick-facts label "Nacionalidad" with value "Americano". Breadcrumb label: "Guionista americano".',
 "https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr": "Nationality: American. Born: Edward Davis Wood, Poughkeepsie, NY, 10 October 1924.",
 "https://themoviedb.org/person/44763-edward-d-wood-jr/translations": "was an American screenwriter, director, producer, actor, author, and editor",
}
ERR = {
 "https://www.entityfacts.com/en/person/scott-derrickson": "HTTP 403 Forbidden",
 "https://wiki.kidzsearch.com/wiki/Ed_Wood": "HTTP 403 Forbidden",
 "https://biola.atlassian.net/wiki/x/C4-p": "empty response (HTTP 202)",
 "https://stream4u.it.com/en/person/55499": "HTTP 404 Not Found",
}

def mk(pairs, nfetch=3):
    out = []
    for i, (t, u) in enumerate(pairs):
        r = {"title": t, "url": u, "snippet": ""}
        if i < nfetch and u in SN:
            r["snippet"] = SN[u]
        elif i < nfetch and u in ERR:
            r["fetch_error"] = ERR[u]
        else:
            r["fetch_error"] = "not_in_top_3"
        out.append(r)
    return out

S = [
 ("hop", 0, "What nationality is Scott Derrickson?", [
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/biografia/"),
  ("Scott Derrickson","https://www.sensacine.com/actores/actor-99042/biografia/"),
  ("scott derrickson","https://www.entityfacts.com/en/person/scott-derrickson"),
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/"),
  ("m.imdb.com","https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson","https://www.ecartelera.com/personas/scott-derrickson/"),
  ("Derrickson","https://en.wikipedia.org/wiki/Derrickson"),
  ("Scott Derrickson","https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  ("Scott Derrickson","https://en.wikipedia.org/wiki/Scott_Derrickson"),
  ("m.imdb.com","https://m.imdb.com/es/name/nm0220600/awards")]),
 ("hop_paraphrase", 0, "Scott Derrickson nationality director", [
  ("Scott Derrickson","https://www.sensacine.com/actores/actor-99042/biografia/"),
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/biografia/"),
  ("scott derrickson","https://www.entityfacts.com/en/person/scott-derrickson"),
  ("Scott Derrickson","https://moviemanner.com/people/scott-derrickson"),
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/"),
  ("m.imdb.com","https://m.imdb.com/name/nm0220600"),
  ("Scott Derrickson","https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  ("scott derrickson","https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  ("Scott Derrickson","https://www.ecartelera.com/personas/scott-derrickson/"),
  ("Scott Derrickson","https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Scott_Derrickson.html")]),
 ("hop", 1, "What nationality was Ed Wood?", [
  ("wood edward d jr","https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr"),
  ("Ed Wood","https://wiki.kidzsearch.com/wiki/Ed_Wood"),
  ("English (en-US)","https://themoviedb.org/person/44763-edward-d-wood-jr/translations"),
  ("By Bill Kleppel","https://poklib.org/ed-wood-b-movie-cult-icon-poughkeepsiean/"),
  ("Edward Wood","https://en.wikipedia.org/wiki/Edward_Wood"),
  ("www.imdb.com","https://www.imdb.com:443/name/nm0000248/mediaviewer/rm1486738176"),
  ("Ed Wood","https://en.wikipedia.org/wiki/Ed_Wood"),
  ("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste","https://www.scifi-universe.com/fiche_personne.asp?per_id=954"),
  ("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste","https://www.scifi-universe.com/personnalites/954/edward-d-wood-jr"),
  ("data.cinemabelgica.be","https://data.cinemabelgica.be/person/9855")]),
 ("hop_paraphrase", 1, "Ed Wood director nationality", [
  ("wood edward d jr","https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr"),
  ("Ed Wood","https://wiki.kidzsearch.com/wiki/Ed_Wood"),
  ("English (en-US)","https://themoviedb.org/person/44763-edward-d-wood-jr/translations"),
  ("data.cinemabelgica.be","https://data.cinemabelgica.be/person/9855"),
  ("Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker,","https://www.acmi.net.au/creators/27635"),
  ("www.imdb.com","https://www.imdb.com:443/title/tt0109707/fullcredits/cast"),
  ("Ed Wood","https://discordia.fandom.com/wiki/Ed_Wood"),
  ("g.co","https://g.co/kg/m/02vt5"),
  ("14 campy facts about ed wood","https://www.mentalfloss.com/entertainment/movies/14-campy-facts-about-ed-wood"),
  ("secure.paperbackswap.com","https://secure.paperbackswap.com/Ed-Wood/author/?t=al")]),
 ("hop", 2, "Scott Derrickson Ed Wood same nationality comparison", [
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/biografia/"),
  ("Scott Derrickson","https://www.sensacine.com/actores/actor-99042/biografia/"),
  ("scott derrickson","https://www.entityfacts.com/en/person/scott-derrickson"),
  ("Scott Derrickson","https://www.sensacine.com.mx/actores/actor-99042/"),
  ("stream4u.it.com","https://stream4u.it.com/en/person/55499"),
  ("Scott Derrickson","https://Www.Wikipedia.org/wiki/Bermuda_(film)"),
  ("m.imdb.com","https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson","https://empireonline.com/people/scott-derrickson"),
  ("scott derrickson","https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  ("movies.oktaycolakoglu.com","https://movies.oktaycolakoglu.com/people/55499")]),
 ("hop_paraphrase", 2, "compare nationalities Scott Derrickson Ed Wood", [
  ("C4 p","https://biola.atlassian.net/wiki/x/C4-p"),
  ("English (en-US)","https://themoviedb.org/person/44763-edward-d-wood-jr/translations"),
  ("stream4u.it.com","https://stream4u.it.com/en/person/55499"),
  ("Scott Derrickson","https://www.sensacine.com/actores/actor-99042/biografia/"),
  ("Scott Derrickson","https://biola.atlassian.net/wiki/x/34Lp"),
  ("m.imdb.com","https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson","https://empireonline.com/people/scott-derrickson"),
  ("Scott Derrickson","https://m.the-numbers.com/people/directors/0SCDC.php"),
  ("m.imdb.com","https://m.imdb.com/name/nm0220600"),
  ("Scott Derrickson","https://streamin.co/ee/person/scott-derrickson-55499")]),
]

have = {(e["qid"], e["search_type"], e["hop_idx"]) for e in data}
for st, hi, q, pairs in S:
    if ("sample_3hop_c", st, hi) in have:
        continue
    data.append({"qid": "sample_3hop_c", "search_type": st, "hop_idx": hi, "query_text": q,
                 "results": mk(pairs),
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    json.dump(data, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(data))
