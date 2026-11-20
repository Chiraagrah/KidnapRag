import json, sys, datetime
P = sys.argv[1]
d = json.load(open(P, encoding="utf-8"))
have = {(e["qid"], e["search_type"], e["hop_idx"]) for e in d}
Q = "sample_3hop_c"


def R(title, url, snip=None, err=None):
    r = {"title": title, "url": url, "snippet": snip or ""}
    if not snip:
        r["fetch_error"] = err or "not fetched (beyond top 3)"
    return r


S1 = "https://www.sensacine.com.mx/actores/actor-99042/biografia/"
S2 = "https://www.sensacine.com/actores/actor-99042/biografia/"
EF = "https://www.entityfacts.com/en/person/scott-derrickson"
ENC = "https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr"
KZ = "https://wiki.kidzsearch.com/wiki/Ed_Wood"
TM = "https://themoviedb.org/person/44763-edward-d-wood-jr/translations"
sm = "Quick-facts box lists nationality as 'Americano'. Biography: 'nació el 16 de julio de 1966 en Denver, Colorado, Estados Unidos'; studied at Biola University and the University of Southern California."
ss = "Quick-facts box lists nationality as 'Americano' (breadcrumb: 'Guionista americano'); activities: Guionista, Director, Productor delegado. Degree in Humanities from Biola University, master's in film production from USC; lives in Glendale, California."
we = "Nationality: American. Born: Edward Davis Wood, Poughkeepsie, NY, 10 October 1924. Served in the U.S. Marine Corps during World War II before journeying to Hollywood."
wt = "English biography (from Wikipedia): Ed Wood 'was an American screenwriter, director, producer, actor, author, and editor' (October 10, 1924 - December 10, 1978)."
F403 = "HTTP 403 Forbidden"

entries = [
 ("hop", 0, "What nationality is Scott Derrickson?", [
  R("Scott Derrickson", S1, sm), R("Scott Derrickson", S2, ss), R("scott derrickson", EF, err=F403),
  R("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  R("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  R("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
  R("Derrickson", "https://en.wikipedia.org/wiki/Derrickson"),
  R("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  R("Scott Derrickson", "https://en.wikipedia.org/wiki/Scott_Derrickson"),
  R("m.imdb.com", "https://m.imdb.com/es/name/nm0220600/awards")]),
 ("hop_paraphrase", 0, "Scott Derrickson nationality director", [
  R("Scott Derrickson", S2, "Profile lists activities 'Guionista, Director, Productor delegado' and nationality 'Americano'. Gives birth date 18 de marzo de 1977, age 49."),
  R("Scott Derrickson", S1, "Also lists nationality 'Americano'; known for directing Doctor Strange (2016). Birth given as 16 de julio de 1966 in Denver, Colorado."),
  R("scott derrickson", EF, err=F403),
  R("Scott Derrickson", "https://moviemanner.com/people/scott-derrickson"),
  R("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  R("m.imdb.com", "https://m.imdb.com/name/nm0220600"),
  R("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  R("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  R("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
  R("Scott Derrickson", "https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Scott_Derrickson.html", err="skipped_low_value_domain")]),
 ("hop", 1, "What nationality was Ed Wood?", [
  R("wood edward d jr", ENC, we), R("Ed Wood", KZ, err=F403), R("English (en-US)", TM, wt),
  R("By Bill Kleppel", "https://poklib.org/ed-wood-b-movie-cult-icon-poughkeepsiean/"),
  R("Edward Wood", "https://en.wikipedia.org/wiki/Edward_Wood"),
  R("www.imdb.com", "https://www.imdb.com:443/name/nm0000248/mediaviewer/rm1486738176"),
  R("Ed Wood", "https://en.wikipedia.org/wiki/Ed_Wood"),
  R("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste", "https://www.scifi-universe.com/fiche_personne.asp?per_id=954"),
  R("Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste", "https://www.scifi-universe.com/personnalites/954/edward-d-wood-jr"),
  R("By Bill Kleppel", "https://poklib.org/?p=38792")]),
 ("hop_paraphrase", 1, "Ed Wood director nationality", [
  R("wood edward d jr", ENC, we), R("Ed Wood", KZ, err=F403),
  R("English (en-US)", TM, wt + " The German translation also describes him as an American screenwriter and director."),
  R("data.cinemabelgica.be", "https://data.cinemabelgica.be/person/9855"),
  R("Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker,", "https://www.acmi.net.au/creators/27635"),
  R("www.imdb.com", "https://www.imdb.com:443/title/tt0109707/fullcredits/cast"),
  R("Ed Wood", "https://discordia.fandom.com/wiki/Ed_Wood"),
  R("g.co", "https://g.co/kg/m/02vt5"),
  R("14 campy facts about ed wood", "https://www.mentalfloss.com/entertainment/movies/14-campy-facts-about-ed-wood"),
  R("secure.paperbackswap.com", "https://secure.paperbackswap.com/Ed-Wood/author/?t=al")]),
 ("hop", 2, "Scott Derrickson Ed Wood same nationality comparison", [
  R("Scott Derrickson", S1, "Quick-facts 'Nacionalidad': 'Americano'. 'El cineasta, Scott Derrickson, nació el 16 de julio de 1966 en Denver, Colorado, Estados Unidos.' Studied Humanities at Biola University. Ed Wood not mentioned."),
  R("Scott Derrickson", S2, "Quick-facts 'Nacionalidad': 'Americano'. Humanities degree from Biola University; worked with Wim Wenders on the story of Land of Plenty. Birth date given as 18 March 1977. Ed Wood not mentioned."),
  R("scott derrickson", EF, err=F403),
  R("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  R("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
  R("Scott Derrickson", "https://Www.Wikipedia.org/wiki/Bermuda_(film)"),
  R("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  R("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
  R("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  R("movies.oktaycolakoglu.com", "https://movies.oktaycolakoglu.com/people/55499")]),
 ("hop_paraphrase", 2, "compare nationalities Scott Derrickson Ed Wood", [
  R("C4 p", "https://biola.atlassian.net/wiki/x/C4-p", err="empty response body (HTTP 202)"),
  R("English (en-US)", TM, "English biography calls Ed Wood (born October 10, 1924; died December 10, 1978) 'an American screenwriter, director, producer, actor, author, and editor'. Ukrainian translation also says American. Page does not mention Scott Derrickson."),
  R("stream4u.it.com", "https://stream4u.it.com/en/person/55499", err="HTTP 404 Not Found"),
  R("Scott Derrickson", S2),
  R("Scott Derrickson", "https://biola.atlassian.net/wiki/x/34Lp"),
  R("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  R("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
  R("Scott Derrickson", "https://m.the-numbers.com/people/directors/0SCDC.php"),
  R("m.imdb.com", "https://m.imdb.com/name/nm0220600"),
  R("Scott Derrickson", "https://streamin.co/ee/person/scott-derrickson-55499")]),
]

for st, hi, q, res in entries:
    if (Q, st, hi) in have:
        continue
    d.append({"qid": Q, "search_type": st, "hop_idx": hi, "query_text": q, "results": res,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
