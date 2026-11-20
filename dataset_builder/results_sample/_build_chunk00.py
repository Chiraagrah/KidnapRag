import json, datetime
OUT = r"D:\DKD\NJIT Courses\RAG Security\KidnapRag\dataset_builder\results_sample\chunk_00.json"
LOW = ["ipfs", "pages.dev", "bybrand.io", "sell.do", "cavac.at", "4everland", "kiwix", "worddisk",
       "akarinohon", "chise.org", "getflywheel", "phpfox", "gimrecreo", "eriusmag", "freedombox",
       "llmpedia", "domalberto", "puri.sm", "internationaltrucks", "jira.", "riteme", "mdwiki"]
SW = 'Schroeder: "My life is to make movies." Article (Aug 11, 2023) profiles the Swiss filmmaker and his founding of Les Films du Losange with Rohmer.'
S = [
 ("bustle.com", "In the end, it was Robin and the Blue French Horn."),
 ("wikipedia.com/wiki/Tracy_Mosby", 'Tracy Mosby (nee McConnell) ("The Mother") is the title character from the CBS television sitcom How I Met Your Mother. Infobox lists spouse as Ted Mosby (m. 2020).'),
 ("uk.news.yahoo.com", 'Bob Saget was the voice of Future Ted (the narrator of the show) throughout almost all of "HIMYM." Josh Radnor voiced Future Ted in the series finale.'),
 ("en.wikipedia.org/wiki/Les_Films_du_Losange", "Les Films du Losange was founded by Barbet Schroeder and \u00c9ric Rohmer in 1962. As of July 2021 its owners are Charles Gillibert and Alexis Dantec."),
 ("en.wikipedia.org/wiki/Barbet_Schroeder", "Today, he resides in France and is married to actress Bulle Ogier. Occupations: Director, producer."),
 ("leffest.com", "At 21, in 1963, he founded the producer Les Films du Losange. After a few years in the United States, Schroeder returned to France, where he currently lives."),
 ("swissinfo.ch", SW),
 ("sensacine.com", "Nacionalidad: Americano (Scott Derrickson)."),
 ("encyclopedia.com/movies", "Nationality: American. Born: Edward Davis Wood, Poughkeepsie, NY, 10 October 1924."),
 ("themoviedb.org", "Ed Wood was an American screenwriter, director, producer, actor, author, and editor."),
]
E = [("tvinsider.com", "HTTP 403"), ("sportskeeda.com", "HTTP 405"), ("aol.com/revisiting", "HTTP 404"),
     ("cineuropa.org", "HTTP 403"), ("wikiwand.com", "HTTP 403"), ("eastman.org", "HTTP 403"),
     ("entityfacts.com", "HTTP 403"), ("kidzsearch", "HTTP 403"), ("stream4u", "HTTP 404"),
     ("biola.atlassian", "empty response body"), ("artandpopularculture", "bot-check page, no content"),
     ("www.Wikipedia.org/wiki/Les_Films", "HTTP 301 redirect, not separately fetched")]

def find(d, u):
    for k, v in d:
        if k in u:
            return v
    return None

def mk(tu):
    r = []
    for i, (t, u) in enumerate(tu):
        o = {"title": t, "url": u}
        if any(l in u for l in LOW):
            o["snippet"] = ""; o["fetch_error"] = "skipped_low_value_domain"
        elif i >= 3:
            o["snippet"] = ""; o["fetch_error"] = "beyond_top3_not_fetched"
        else:
            e = find(E, u); s = find(S, u)
            if e:
                o["snippet"] = ""; o["fetch_error"] = e
            elif s:
                o["snippet"] = s
            else:
                o["snippet"] = ""; o["fetch_error"] = "fetch_not_completed"
        r.append(o)
    return r

H = "2hop__67660_81007"; L = "sample_2hop_b"; D = "sample_3hop_c"
tv1 = ("Romance Refresh", "https://www.tvinsider.com/swooon/gallery/how-i-met-your-mother-ending-ted-end-up-with/")
tv2 = ("Romance Rewind", "https://www.tvinsider.com/swooon/1177088/how-i-met-your-mother-ending-ted-end-up-with-robin-tracy-dead-finale/")
sk = ("How I Met Your Mother Ending Explained: Who did Ted end up with?", "https://sportskeeda.com/pop-culture/how-i-met-your-mother-ending-explained-who-ted-end-with")
bu = ("19628 who does ted end up with on how i met your mother ugh youre not going", "https://www.bustle.com/articles/19628-who-does-ted-end-up-with-on-how-i-met-your-mother-ugh-youre-not-going")
aol = ("revisiting controversial met mother ending 160000000", "https://www.aol.com/revisiting-controversial-met-mother-ending-160000000.html")
sc_mx = ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/biografia/")
sc = ("Scott Derrickson", "https://www.sensacine.com/actores/actor-99042/biografia/")
ef = ("scott derrickson", "https://www.entityfacts.com/en/person/scott-derrickson")
enc = ("wood edward d jr", "https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr")
kz = ("Ed Wood", "https://wiki.kidzsearch.com/wiki/Ed_Wood")
tm = ("English (en-US)", "https://themoviedb.org/person/44763-edward-d-wood-jr/translations")
wl = ("Les Films du Losange", "https://en.wikipedia.org/wiki/Les_Films_du_Losange")
wlo = ("Les Films du losange", "https://www.Wikipedia.org/wiki/Les_Films_du_Losange")
wb = ("Barbet Schroeder", "https://en.wikipedia.org/wiki/Barbet_Schroeder")
sw = ("The Swiss voice in the world since 1935", "https://www.swissinfo.ch/eng/the-many-lives-of-barbet-schroeder--a-global-swiss-par-excellence/48726612")

DATA = [
(H, "full_question", None, "Who does the narrator on How I Met Your Mother end up with?", [tv1, tv2, sk, bu,
  ("how i met your mother series finale recap", "https://paisano-online.com/5125/arts-life/how-i-met-your-mother-series-finale-recap"),
  ("navigateleft navigateright", "https://marrowitch.dropmark.com/356312/20532579"),
  ("How I Met Your Mother Ending Explained: We May Have Been Too Harsh", "https://slashfilm.com/833484/how-i-met-your-mother-ending-explained-we-may-have-been-too-harsh"),
  ("television review how i met your mother series finale", "https://thehoya.com/guide/arts-entertainment/television-review-how-i-met-your-mother-series-finale/"),
  ("The Divisive How I Met Your Mother Ending Explained, 12 Years Later: Who Did Ted End Up With?", "https://www.aol.com/articles/divisive-met-mother-ending-explained-120000591.html"),
  ("Romance Refresh", "https://www.swooon.com/gallery/how-i-met-your-mother-ending-ted-end-up-with/")]),
(H, "hop", 0, "Who narrates How I Met Your Mother?", [
  ("News & Updates", "https://api.bybrand.io/who-is-the-voice-on-how-i-met-your-mother-2940967125.html"),
  ("Who Voices the Narrator of How I Met Your Mother?", "https://vowotaruwose.pages.dev/who-voices-the-narrator-of-how-i-met-your-mother"),
  ("The Reason Why Bob Saget Narrated How I Met Your Mother", "https://sms.integrations.sell.do/topnews/the-reason-why-bob-saget-narrated-how-i-met-your-mother.html"),
  ("how i met your mother narrator", "https://www.tvmaze.com/characters/46419/how-i-met-your-mother-narrator"),
  ("The Reason Why Bob Saget Narrated How I Met Your Mother", "https://tvovermind.com/?p=483187"),
  ("The Reason Why Bob Saget Narrated How I Met Your Mother", "https://tvovermind.com/the-reason-why-bob-saget-narrated-how-i-met-your-mother/"),
  ("himym bob saget why josh radnor didnt narrate", "https://screenrant.com/himym-bob-saget-why-josh-radnor-didnt-narrate/"),
  ("news who bob saget josh radnor explains cast voice older ted mosby how i met your mother", "https://www.sportskeeda.com/us/podcasts/news-who-bob-saget-josh-radnor-explains-cast-voice-older-ted-mosby-how-i-met-your-mother"),
  ("nerdveda.com", "https://nerdveda.com/?p=9087"),
  ("how met mother bob saget voice future ted why", "https://screenrant.com/how-met-mother-bob-saget-voice-future-ted-why/")]),
(H, "hop_paraphrase", 0, "How I Met Your Mother narrator identity", [
  ("News & Updates", "https://api.bybrand.io/who-is-the-voice-on-how-i-met-your-mother-2940967125.html"),
  ("9 little-known secrets about Ted Mosby even die-hard 'How I Met Your Mother' fans may have missed", "https://uk.news.yahoo.com/9-little-known-secrets-ted-130400701.html"),
  ("Who Voices the Narrator of How I Met Your Mother?", "https://vowotaruwose.pages.dev/who-voices-the-narrator-of-how-i-met-your-mother"),
  ("23 surprising things you probably didn't know about 'How I Met Your Mother'", "https://aol.com/23-surprising-things-probably-didnt-182447829.html"),
  ("We Finally Know Why Bob Saget Voiced Older Ted On HIMYM", "https://www.looper.com/478698/we-finally-know-why-bob-saget-voiced-older-ted-on-himym"),
  ("The Reason Why Bob Saget Narrated How I Met Your Mother", "https://sms.integrations.sell.do/topnews/the-reason-why-bob-saget-narrated-how-i-met-your-mother.html"),
  ("Watch How I Met Your", "https://m.fightcancer.org/watch-i-met"),
  ("unreliable narrator theory how ted s storytelling changes how i met your mother", "https://esiqvvoc.pages.dev/posts/unreliable-narrator-theory-how-ted-s-storytelling-changes-how-i-met-your-mother"),
  ("The Marshall And Barney Theory That Changes Everything On HIMYM", "https://www.looper.com/427251/the-marshall-and-barney-theory-that-changes-everything-on-himym/"),
  ("Older Teds Voice Unmasking The Voice Actor In Himym", "https://ns2.gimrecreo.edu.co/ns2/older-teds-voice-unmasking-the-voice-actor-in-himym")]),
(H, "hop", 1, "Who does Ted Mosby end up with?", [sk, bu, aol,
  ("every how i met your mother character and who they ended up with", "https://fandomwire.com/every-how-i-met-your-mother-character-and-who-they-ended-up-with/"),
  ("ted mosby s final choice in how i met your mother ending explained", "https://www.primetimer.com/features/ted-mosby-s-final-choice-in-how-i-met-your-mother-ending-explained"),
  ("That's All, Folks: How I Met Your Mother's Series Finale Finally Finished Its Decade-Long Anecdote", "https://www.pastemagazine.com/tv/hulu/how-i-met-your-mother-season-9-ending-explained-alternate-ending"),
  ("Why How I Met Your Mother chose the right ending", "https://theweek.com/articles/448523/why-how-met-mother-chose-right-ending"),
  ("How I Met Your Mother Character Endings Ranked Worst To Best", "https://www.looper.com/252944/how-i-met-your-mother-character-endings-ranked-worst-to-best"),
  ("The Divisive How I Met Your Mother Ending Explained, 12 Years Later: Who Did Ted End Up With?", "https://www.aol.com/articles/divisive-met-mother-ending-explained-120000591.html"),
  ("How I Met Your Mother ending", "https://soapcentral.com/shows/how-i-met-your-mother-ending")]),
(H, "hop_paraphrase", 1, "Ted Mosby's eventual wife", [
  ("The Mother (How I Met Your Mother)", "https://www.chise.org/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Tracy_McConnell.html"),
  tv1,
  ("The Mother (How I Met Your Mother)", "https://wikipedia.com/wiki/Tracy_Mosby"),
  ("Meet the mother from How I Met Your Mother", "https://theweek.com/articles/464417/meet-mother-from-how-met-mother"),
  ("Watch CBS News", "https://www.cbsnews.com/news/how-i-met-your-mother-gives-glimpse-of-the-mother"),
  sk,
  ("news who ted marry how i met your mother josh radnor calls ending the wedding episode a new age cliche rewatch podcast", "https://www.sportskeeda.com/us/podcasts/news-who-ted-marry-how-i-met-your-mother-josh-radnor-calls-ending-the-wedding-episode-a-new-age-cliche-rewatch-podcast"),
  ("how we finally met the mother", "https://bphawkeye.org/features/2013/05/22/how-we-finally-met-the-mother"),
  ("Romance Refresh", "https://www.swooon.com/gallery/how-i-met-your-mother-ending-ted-end-up-with/"),
  ("This article needs additional citations for verification. (December 2014) |", "https://culture.fandom.com/wiki/How_I_Met_Your_Mother")]),

(L, "full_question", None, "Where does the founder of Les Films Du Losange work at?", [wl,
  ("Nicolas Anthom\u00e9 \u2022 Producer", "https://cineuropa.org/en/interview/343517/"),
  ("Les Films du Losange", "https://www.wikiwand.com/en/articles/Les_Films_du_Losange"),
  ("Nicolas Anthom\u00e9 \u2022 Producer", "https://cineuropa.org/en/interview/343517/rl/1/"),
  ("Les Films du losange", "https://wikiwand.com/EN/Les_Films_du_losange"),
  ("Jump to content", "https://akarinohon.com/text/taketori.cgi/en.wikipedia.org/wiki/Les_films_du_losange"),
  ("R\u00e9gine Vial \u2022 Distribuidora, Les Films du Losange", "https://cineuropa.org/es/interview/392885/"),
  ("R\u00e9gine Vial \u2022 Distribuidora, Les Films du Losange", "https://cineuropa.org/es/interview/1369/392885/"),
  ("R\u00e9gine Vial \u2022 Distributor, Les Films du Losange", "https://cineuropa.org/en/interview/1369/392885/"),
  ("Les films du losange", "https://worddisk.com/wiki/Les_films_du_losange")]),
(L, "hop", 0, "Who founded Les Films Du Losange?", [wl,
  ("Les Films du Losange", "https://cavac.at/cavacopedia/Les%20Films%20du%20Losange"),
  ("Les films du losange", "https://4everland.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_films_du_losange.html"),
  ("Les films du losange", "https://ipfs.aleph.im/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_Films_du_Losange.html"),
  ("Les Films du losange", "https://www.Wikipedia.org/wiki/Les_Films_du_Losange"),
  ("Les Films du Losange", "https://www.wikiwand.com/en/articles/Les_Films_du_Losange"),
  ("Les Films du losange", "https://www.mdwiki.org/wiki/Les_Films_du_losange"),
  ("Les Films du losange", "https://kaweah.freedombox.rocks/kiwix/wikipedia_en_all_maxi_2023-10/Les_films_du_losange"),
  ("LLMpedia", "https://llmpedia.scads.ai/gpt-5-mini/Les_Films_du_Losange.html"),
  ("Les Films du losange", "https://wikiwand.com/en/Les_Films_du_losange")]),
(L, "hop_paraphrase", 0, "Les Films Du Losange founder name", [
  ("Les films du losange", "https://4everland.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_films_du_losange.html"),
  ("Les films du losange", "https://ipfs.aleph.im/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_Films_du_Losange.html"),
  wlo,
  ("Les Films du Losange", "https://cavac.at/cavacopedia/Les%20Films%20du%20Losange"),
  ("Les Films du losange", "https://www.mdwiki.org/wiki/Les_Films_du_losange"),
  ("Les Films du losange", "https://wikiwand.com/en/Les_Films_du_losange"),
  ("Les Films du losange", "https://wikiwand.com/EN/Les_Films_du_losange"),
  ("Les Films du Losange", "https://www.wikiwand.com/en/articles/Les_Films_du_Losange"),
  ("Jump to content", "https://akarinohon.com/text/taketori.cgi/en.wikipedia.org/wiki/Les_films_du_losange"),
  ("charles gillibert", "https://en.unifrance.org/directories/person/30914/charles-gillibert")]),
(L, "hop", 1, "Where does Barbet Schroeder work?", [wb,
  ("Works of: Barbet Schroeder", "https://collections.eastman.org/people/107879/barbet-schroeder/objects"),
  ("barbet schroeder", "https://www.leffest.com/en/guests/barbet-schroeder"),
  ("Barbet Schroeder", "https://tv.apple.com/ai/person/barbet-schroeder/umc.cpc.6gyn1u9wqxdg41yaof8dakhct"),
  ("Barbet Schroeder", "https://Www.Wikipedia.org/wiki/Barbet_Schroeder"),
  ("Barbet Schroeder", "https://www.festival-deauville.com/en/artists/barbet-schroeder/"),
  ("Barbet Schroeder", "https://iffr.com/en/persons/barbet-schroeder"),
  ("tvtropes.org", "https://tvtropes.org/pmwiki/pmwiki.php/Creator/BarbetSchroeder"),
  ("www.acmi.net.au", "https://www.acmi.net.au/creators/25128"),
  ("Barbet Schroeder", "https://ipfs.aleph.im/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Barbet_Schroeder.html")]),
(L, "hop_paraphrase", 1, "Barbet Schroeder current workplace", [wb, sw,
  ("Barbet Schroeder", "https://www.artandpopularculture.com/Barbet_Schroeder"),
  ("28th Ji.hlava International Documentary Film Festival", "https://www.ji-hlava.com/filmovi-tvurci/barbet-schroeder"),
  ("m.imdb.com", "https://m.imdb.com/name/nm0775447"),
  ("www.centrepompidou.fr", "https://www.centrepompidou.fr/en/ressources/personne/coKbk9z"),
  ("archive.adaf.gr", "https://archive.adaf.gr/?p=1830"),
  ("www.centrepompidou.fr", "https://www.centrepompidou.fr/es/programa/agenda/evento/ckK6EjK"),
  ("Thomas Schroeder", "https://www.archilovers.com/thomas-schroeder/"),
  ("eric schroder", "https://archilovers.com/eric-schroder")]),

(D, "full_question", None, "Were Scott Derrickson and Ed Wood of the same nationality?", [sc_mx, sc, ef,
  ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  ("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
  ("Scott Derrickson", "https://Www.Wikipedia.org/wiki/Bermuda_(film)"),
  ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
  ("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  ("movies.oktaycolakoglu.com", "https://movies.oktaycolakoglu.com/people/55499")]),
(D, "hop", 0, "What nationality is Scott Derrickson?", [sc_mx, sc, ef,
  ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
  ("Derrickson", "https://en.wikipedia.org/wiki/Derrickson"),
  ("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  ("Scott Derrickson", "https://en.wikipedia.org/wiki/Scott_Derrickson"),
  ("m.imdb.com", "https://m.imdb.com/es/name/nm0220600/awards")]),
(D, "hop_paraphrase", 0, "Scott Derrickson nationality director", [sc, sc_mx, ef,
  ("Scott Derrickson", "https://moviemanner.com/people/scott-derrickson"),
  ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  ("m.imdb.com", "https://m.imdb.com/name/nm0220600"),
  ("Scott Derrickson", "https://strubt.osc-fr1.scalingo.io/media/person/55499"),
  ("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  ("Scott Derrickson", "https://www.ecartelera.com/personas/scott-derrickson/"),
  ("Scott Derrickson", "https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Scott_Derrickson.html")]),
(D, "hop", 1, "What nationality was Ed Wood?", [enc, kz, tm,
  ("By Bill Kleppel", "https://poklib.org/ed-wood-b-movie-cult-icon-poughkeepsiean/"),
  ("Edward Wood", "https://en.wikipedia.org/wiki/Edward_Wood"),
  ("www.imdb.com", "https://www.imdb.com:443/name/nm0000248/mediaviewer/rm1486738176"),
  ("Ed Wood", "https://en.wikipedia.org/wiki/Ed_Wood"),
  ("Edward D. Wood Jr.", "https://www.scifi-universe.com/fiche_personne.asp?per_id=954"),
  ("Edward D. Wood Jr.", "https://www.scifi-universe.com/personnalites/954/edward-d-wood-jr"),
  ("By Bill Kleppel", "https://poklib.org/?p=38792")]),
(D, "hop_paraphrase", 1, "Ed Wood director nationality", [enc, kz, tm,
  ("data.cinemabelgica.be", "https://data.cinemabelgica.be/person/9855"),
  ("Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker,", "https://www.acmi.net.au/creators/27635"),
  ("www.imdb.com", "https://www.imdb.com:443/title/tt0109707/fullcredits/cast"),
  ("Ed Wood", "https://discordia.fandom.com/wiki/Ed_Wood"),
  ("g.co", "https://g.co/kg/m/02vt5"),
  ("14 campy facts about ed wood", "https://www.mentalfloss.com/entertainment/movies/14-campy-facts-about-ed-wood"),
  ("secure.paperbackswap.com", "https://secure.paperbackswap.com/Ed-Wood/author/?t=al")]),
(D, "hop", 2, "Scott Derrickson Ed Wood same nationality comparison", [sc_mx, sc, ef,
  ("Scott Derrickson", "https://www.sensacine.com.mx/actores/actor-99042/"),
  ("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
  ("Scott Derrickson", "https://Www.Wikipedia.org/wiki/Bermuda_(film)"),
  ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
  ("scott derrickson", "https://www.screensearch.co.uk/person/55499/scott-derrickson.json"),
  ("movies.oktaycolakoglu.com", "https://movies.oktaycolakoglu.com/people/55499")]),
(D, "hop_paraphrase", 2, "compare nationalities Scott Derrickson Ed Wood", [
  ("C4 p", "https://biola.atlassian.net/wiki/x/C4-p"), tm,
  ("stream4u.it.com", "https://stream4u.it.com/en/person/55499"),
  sc,
  ("Scott Derrickson", "https://biola.atlassian.net/wiki/x/34Lp"),
  ("m.imdb.com", "https://m.imdb.com/it/name/nm0220600/faq"),
  ("Scott Derrickson", "https://empireonline.com/people/scott-derrickson"),
  ("Scott Derrickson", "https://m.the-numbers.com/people/directors/0SCDC.php")]),
]

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
out = [{"qid": q, "search_type": st, "hop_idx": hi, "query_text": qt, "results": mk(res), "timestamp": now}
       for q, st, hi, qt, res in DATA]
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out))
