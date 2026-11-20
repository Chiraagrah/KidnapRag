import json, datetime
OUT = r"D:\DKD\NJIT Courses\RAG Security\KidnapRag\dataset_builder\results_sample\chunk_00.json"
JUNK = ["ipfs", "bybrand", "pages.dev", "sell.do", "phpfox", "4everland"]

def L(s):
    return [tuple(x.split(" || ")) for x in s.strip().split("\n")]

R1 = L("""Romance Refresh || https://www.tvinsider.com/swooon/gallery/how-i-met-your-mother-ending-ted-end-up-with/
Romance Rewind || https://www.tvinsider.com/swooon/1177088/how-i-met-your-mother-ending-ted-end-up-with-robin-tracy-dead-finale/
How I Met Your Mother Ending Explained: Who did Ted end up with? || https://sportskeeda.com/pop-culture/how-i-met-your-mother-ending-explained-who-ted-end-with
how i met your mother series finale recap || https://paisano-online.com/5125/arts-life/how-i-met-your-mother-series-finale-recap
19628 who does ted end up with on how i met your mother ugh youre not going || https://www.bustle.com/articles/19628-who-does-ted-end-up-with-on-how-i-met-your-mother-ugh-youre-not-going
navigateleft navigateright || https://marrowitch.dropmark.com/356312/20532579
Saleha Irfan, Senior Sub-Editor/Reporter || https://gulftoday.ae/Culture/2019/10/27/Did-How-I-Met-Your-Mother-tease-its-ending-in-the-first-season
How I Met Your Mother Ending Explained: We May Have Been Too Harsh || https://slashfilm.com/833484/how-i-met-your-mother-ending-explained-we-may-have-been-too-harsh
television review how i met your mother series finale || https://thehoya.com/guide/arts-entertainment/television-review-how-i-met-your-mother-series-finale/
Revisiting the Great 'How I Met Your Mother' Debate: Should Ted Have Ended Up With Robin? || https://www.yahoo.com/entertainment/revisiting-great-met-mother-debate-161038637.html""")

R2 = L("""News & Updates || https://api.bybrand.io/who-is-the-voice-on-how-i-met-your-mother-2940967125.html
Who Voices the Narrator of How I Met Your Mother? || https://vowotaruwose.pages.dev/who-voices-the-narrator-of-how-i-met-your-mother
The Reason Why Bob Saget Narrated How I Met Your Mother || https://sms.integrations.sell.do/topnews/the-reason-why-bob-saget-narrated-how-i-met-your-mother.html
how i met your mother narrator || https://www.tvmaze.com/characters/46419/how-i-met-your-mother-narrator
The Reason Why Bob Saget Narrated How I Met Your Mother || https://tvovermind.com/?p=483187
The Reason Why Bob Saget Narrated How I Met Your Mother || https://tvovermind.com/the-reason-why-bob-saget-narrated-how-i-met-your-mother/
himym bob saget why josh radnor didnt narrate || https://screenrant.com/himym-bob-saget-why-josh-radnor-didnt-narrate/
news who bob saget josh radnor explains cast voice older ted mosby how i met your mother || https://www.sportskeeda.com/us/podcasts/news-who-bob-saget-josh-radnor-explains-cast-voice-older-ted-mosby-how-i-met-your-mother
nerdveda.com || https://nerdveda.com/?p=9087
how met mother bob saget voice future ted why || https://screenrant.com/how-met-mother-bob-saget-voice-future-ted-why/""")

R3 = L("""News & Updates || https://api.bybrand.io/who-is-the-voice-on-how-i-met-your-mother-2940967125.html
9 little-known secrets about Ted Mosby even die-hard 'How I Met Your Mother' fans may have missed || https://uk.news.yahoo.com/9-little-known-secrets-ted-130400701.html
Who Voices the Narrator of How I Met Your Mother? || https://vowotaruwose.pages.dev/who-voices-the-narrator-of-how-i-met-your-mother
23 surprising things you probably didn't know about 'How I Met Your Mother' || https://aol.com/23-surprising-things-probably-didnt-182447829.html
We Finally Know Why Bob Saget Voiced Older Ted On HIMYM || https://www.looper.com/478698/we-finally-know-why-bob-saget-voiced-older-ted-on-himym
The Reason Why Bob Saget Narrated How I Met Your Mother || https://sms.integrations.sell.do/topnews/the-reason-why-bob-saget-narrated-how-i-met-your-mother.html
Watch How I Met Your || https://m.fightcancer.org/watch-i-met
unreliable narrator theory how ted s storytelling changes how i met your mother || https://esiqvvoc.pages.dev/posts/unreliable-narrator-theory-how-ted-s-storytelling-changes-how-i-met-your-mother
The Marshall And Barney Theory That Changes Everything On HIMYM || https://www.looper.com/427251/the-marshall-and-barney-theory-that-changes-everything-on-himym/
Older Teds Voice Unmasking The Voice Actor In Himym || https://ns2.gimrecreo.edu.co/ns2/older-teds-voice-unmasking-the-voice-actor-in-himym""")

R4 = L("""How I Met Your Mother Ending Explained: Who did Ted end up with? || https://sportskeeda.com/pop-culture/how-i-met-your-mother-ending-explained-who-ted-end-with
19628 who does ted end up with on how i met your mother ugh youre not going || https://www.bustle.com/articles/19628-who-does-ted-end-up-with-on-how-i-met-your-mother-ugh-youre-not-going
revisiting controversial met mother ending 160000000 || https://www.aol.com/revisiting-controversial-met-mother-ending-160000000.html
every how i met your mother character and who they ended up with || https://fandomwire.com/every-how-i-met-your-mother-character-and-who-they-ended-up-with/
ted mosby s final choice in how i met your mother ending explained || https://www.primetimer.com/features/ted-mosby-s-final-choice-in-how-i-met-your-mother-ending-explained
That's All, Folks: How I Met Your Mother's Series Finale Finally Finished Its Decade-Long Anecdote || https://www.pastemagazine.com/tv/hulu/how-i-met-your-mother-season-9-ending-explained-alternate-ending
Why How I Met Your Mother chose the right ending || https://theweek.com/articles/448523/why-how-met-mother-chose-right-ending
How I Met Your Mother Character Endings Ranked Worst To Best || https://www.looper.com/252944/how-i-met-your-mother-character-endings-ranked-worst-to-best
The Divisive "How I Met Your Mother" Ending Explained, 12 Years Later: Who Did Ted End Up With? || https://www.aol.com/articles/divisive-met-mother-ending-explained-120000591.html
How I Met Your Mother ending || https://soapcentral.com/shows/how-i-met-your-mother-ending""")

R5 = L("""A 'How I Met Your Mother' Spinoff Is on the Way — but How Did the Original End? || https://jira.phpfox.com/p/how-i-met-your-mother-ending
Romance Refresh || https://www.tvinsider.com/swooon/gallery/how-i-met-your-mother-ending-ted-end-up-with/
The Mother (How I Met Your Mother) || https://www.chise.org/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Tracy_McConnell.html
The Untold Truth: 7 Shocking Facts About Tracy McConnell, The Mother From How I Met Your Mother || https://slbubble.studentbridge.com/celebnews031/the-untold-truth-7-shocking-facts-about-tracy-mcconnell-the-mother-from-how-i-met-your-mother-vd8cq.html
Romance Refresh || https://www.swooon.com/gallery/how-i-met-your-mother-ending-ted-end-up-with/
How I Met Your Mother Ending Explained: Who did Ted end up with? || https://sportskeeda.com/pop-culture/how-i-met-your-mother-ending-explained-who-ted-end-with
news who ted marry how i met your mother josh radnor calls ending the wedding episode a new age cliche rewatch podcast || https://www.sportskeeda.com/us/podcasts/news-who-ted-marry-how-i-met-your-mother-josh-radnor-calls-ending-the-wedding-episode-a-new-age-cliche-rewatch-podcast
'How I Met Your Mother' Series Finale Recap: 2 Marriages, A Divorce and Rekindled Love (SPOILERS) || https://www.christianpost.com/trends/how-i-met-your-mother-series-finale-recap-2-marriages-a-divorce-and-rekindled-love-spoilers.html
Who Played Ted's Wife On HIMYM? || https://partnerwithscphealth-jitsi.bluetext.com/blog/who-played-teds-wife-on
Continued from "Last Forever, Part 1" || https://www.tvtropes.org/pmwiki/pmwiki.php/Recap/HowIMetYourMotherS9E24LastForeverPartTwo""")

LOS = L("""Les films du losange || https://4everland.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_films_du_losange.html
Les films du losange || https://ipfs.aleph.im/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Les_Films_du_Losange.html
Les Films du losange || https://www.Wikipedia.org/wiki/Les_Films_du_Losange
Les Films du Losange || https://cavac.at/cavacopedia/Les%20Films%20du%20Losange
Les Films du losange || https://www.mdwiki.org/wiki/Les_Films_du_losange
Les Films du losange || https://wikiwand.com/en/Les_Films_du_losange
Les Films du losange || https://wikiwand.com/EN/Les_Films_du_losange
Les Films du Losange || https://www.wikiwand.com/en/articles/Les_Films_du_Losange
Jump to content || https://akarinohon.com/text/taketori.cgi/en.wikipedia.org/wiki/Les_films_du_losange
charles gillibert || https://en.unifrance.org/directories/person/30914/charles-gillibert
Charles Gillibert || https://wiki.riteme.site/wiki/Gillibert,_Charles
screendaily.com || https://screendaily.com/features/my-screen-life-french-producer-charles-gillibert-on-his-typical-breakfast-and-taking-advice-from-olivier-assayas/5175404.article
les films du losange || https://en.unifrance.org/directories/company/79873/les-films-du-losange
LLMpediaThe first transparent, open encyclopedia generated by LLMs || https://llmpedia.scads.ai/gpt-5-mini/Les_Films_du_Losange.html
TEXTE 60 ANS DU LOSANGE compressed || https://2023.fif-85.com/wp-content/uploads/2023/09/TEXTE-60-ANS-DU-LOSANGE_compressed.pdf
Les films du losange || https://worddisk.com/wiki/Les_films_du_losange
cinando.com || https://cinando.com/en/Company/les_films_du_losange_503/Detail
Les Films du Losange || https://www.allocine.fr/societe/fichesociete-225/""")

S9 = L("""Barbet Schroeder || https://en.wikipedia.org/wiki/Barbet_Schroeder
Works of: Barbet Schroeder || https://collections.eastman.org/people/107879/barbet-schroeder/objects
barbet schroeder || https://www.leffest.com/en/guests/barbet-schroeder
Barbet Schroeder || https://tv.apple.com/ai/person/barbet-schroeder/umc.cpc.6gyn1u9wqxdg41yaof8dakhct
Barbet Schroeder || https://Www.Wikipedia.org/wiki/Barbet_Schroeder
Barbet Schroeder || https://www.festival-deauville.com/en/artists/barbet-schroeder/
Barbet Schroeder || https://iffr.com/en/persons/barbet-schroeder
tvtropes.org || https://tvtropes.org/pmwiki/pmwiki.php/Creator/BarbetSchroeder
www.acmi.net.au || https://www.acmi.net.au/creators/25128
Barbet Schroeder || https://ipfs.aleph.im/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Barbet_Schroeder.html""")

S10 = L("""Barbet Schroeder || https://en.wikipedia.org/wiki/Barbet_Schroeder
Barbet Schroeder || https://iffr.com/en/persons/barbet-schroeder
The Swiss voice in the world since 1935 || https://www.swissinfo.ch/eng/the-many-lives-of-barbet-schroeder--a-global-swiss-par-excellence/48726612
28th Ji.hlava International Documentary Film Festival || https://www.ji-hlava.com/filmovi-tvurci/barbet-schroeder
m.imdb.com || https://m.imdb.com/name/nm0775447
www.centrepompidou.fr || https://www.centrepompidou.fr/en/ressources/personne/coKbk9z
archive.adaf.gr || https://archive.adaf.gr/?p=1830
www.centrepompidou.fr || https://www.centrepompidou.fr/es/programa/agenda/evento/ckK6EjK
Thomas Schroeder || https://www.archilovers.com/thomas-schroeder/
eric schroder || https://archilovers.com/eric-schroder
schroeder laura || https://www.oper.koeln/en/schroeder-laura
Jonathan Schroeder || https://in.marketscreener.com/insider/JONATHAN-SCHROEDER-A2E5SG/
schroeders geneva || https://assets01.sdd1.ch/assets/lbwp-cdn/supravision/files/1765357734/schroeders-geneva.pdf
HERMAN FRIEDRICH SCHROEDER A2K4X8 || https://ch.marketscreener.com/insider/HERMAN-FRIEDRICH-SCHROEDER-A2K4X8/
lynn lecorsais || https://en.paperjam.lu/guide/biography/1072375954/lynn-lecorsais
Jonathan Schroeder || https://es.marketscreener.com/insider/JONATHAN-SCHROEDER-A2E5SG/experiencia/""")

D11 = L("""Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/biografia/
Scott Derrickson || https://www.sensacine.com/actores/actor-99042/biografia/
scott derrickson || https://www.entityfacts.com/en/person/scott-derrickson
Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/
stream4u.it.com || https://stream4u.it.com/en/person/55499
Scott Derrickson || https://Www.Wikipedia.org/wiki/Bermuda_(film)
m.imdb.com || https://m.imdb.com/it/name/nm0220600/faq
Scott Derrickson || https://empireonline.com/people/scott-derrickson
scott derrickson || https://www.screensearch.co.uk/person/55499/scott-derrickson.json
movies.oktaycolakoglu.com || https://movies.oktaycolakoglu.com/people/55499""")

D12 = L("""Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/biografia/
Scott Derrickson || https://www.sensacine.com/actores/actor-99042/biografia/
scott derrickson || https://www.entityfacts.com/en/person/scott-derrickson
Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/
m.imdb.com || https://m.imdb.com/it/name/nm0220600/faq
Scott Derrickson || https://www.ecartelera.com/personas/scott-derrickson/
Derrickson || https://en.wikipedia.org/wiki/Derrickson
Scott Derrickson || https://strubt.osc-fr1.scalingo.io/media/person/55499
Scott Derrickson || https://en.wikipedia.org/wiki/Scott_Derrickson
m.imdb.com || https://m.imdb.com/es/name/nm0220600/awards""")

D13 = L("""Scott Derrickson || https://www.sensacine.com/actores/actor-99042/biografia/
Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/biografia/
scott derrickson || https://www.entityfacts.com/en/person/scott-derrickson
Scott Derrickson || https://moviemanner.com/people/scott-derrickson
Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/
m.imdb.com || https://m.imdb.com/name/nm0220600
Scott Derrickson || https://strubt.osc-fr1.scalingo.io/media/person/55499
scott derrickson || https://www.screensearch.co.uk/person/55499/scott-derrickson.json
Scott Derrickson || https://www.ecartelera.com/personas/scott-derrickson/
Scott Derrickson || https://bafybeiemxf5abjwjbikoz4mc3a3dla6ual3jsgpdr4cjr3oz3evfyavhwq.ipfs.4everland.io/wiki/Scott_Derrickson.html""")

D14 = L("""wood edward d jr || https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr
Ed Wood || https://wiki.kidzsearch.com/wiki/Ed_Wood
English (en-US) || https://themoviedb.org/person/44763-edward-d-wood-jr/translations
By Bill Kleppel || https://poklib.org/ed-wood-b-movie-cult-icon-poughkeepsiean/
Edward Wood || https://en.wikipedia.org/wiki/Edward_Wood
www.imdb.com || https://www.imdb.com:443/name/nm0000248/mediaviewer/rm1486738176
Ed Wood || https://en.wikipedia.org/wiki/Ed_Wood
Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste || https://www.scifi-universe.com/fiche_personne.asp?per_id=954
Edward D. Wood Jr.  Réalisateur, Producteur, Scénariste || https://www.scifi-universe.com/personnalites/954/edward-d-wood-jr
By Bill Kleppel || https://poklib.org/?p=38792""")

D15 = L("""wood edward d jr || https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr
Ed Wood || https://wiki.kidzsearch.com/wiki/Ed_Wood
English (en-US) || https://themoviedb.org/person/44763-edward-d-wood-jr/translations
data.cinemabelgica.be || https://data.cinemabelgica.be/person/9855
Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, || https://www.acmi.net.au/creators/27635
www.imdb.com || https://www.imdb.com:443/title/tt0109707/fullcredits/cast
Ed Wood || https://discordia.fandom.com/wiki/Ed_Wood
g.co || https://g.co/kg/m/02vt5
14 campy facts about ed wood || https://www.mentalfloss.com/entertainment/movies/14-campy-facts-about-ed-wood
secure.paperbackswap.com || https://secure.paperbackswap.com/Ed-Wood/author/?t=al""")

D16 = L("""Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/biografia/
Scott Derrickson || https://www.sensacine.com/actores/actor-99042/biografia/
scott derrickson || https://www.entityfacts.com/en/person/scott-derrickson
Scott Derrickson || https://www.sensacine.com.mx/actores/actor-99042/
stream4u.it.com || https://stream4u.it.com/en/person/55499
Scott Derrickson || https://Www.Wikipedia.org/wiki/Bermuda_(film)
m.imdb.com || https://m.imdb.com/it/name/nm0220600/faq
Scott Derrickson || https://empireonline.com/people/scott-derrickson
scott derrickson || https://www.screensearch.co.uk/person/55499/scott-derrickson.json
movies.oktaycolakoglu.com || https://movies.oktaycolakoglu.com/people/55499""")

D17 = L("""C4 p || https://biola.atlassian.net/wiki/x/C4-p
English (en-US) || https://themoviedb.org/person/44763-edward-d-wood-jr/translations
stream4u.it.com || https://stream4u.it.com/en/person/55499
Scott Derrickson || https://www.sensacine.com/actores/actor-99042/biografia/
Scott Derrickson || https://biola.atlassian.net/wiki/x/34Lp
m.imdb.com || https://m.imdb.com/it/name/nm0220600/faq
Scott Derrickson || https://empireonline.com/people/scott-derrickson
Scott Derrickson || https://m.the-numbers.com/people/directors/0SCDC.php
m.imdb.com || https://m.imdb.com/name/nm0220600
Scott Derrickson || https://streamin.co/ee/person/scott-derrickson-55499""")

# fetched snippets: url -> snippet, or (None, error)
YAHOO = "Bob Saget was the voice of Future Ted (the narrator of the show) throughout almost all of 'HIMYM.' Then, on the finale \"Last Forever, Part 2,\" \"Radnor did double duty as Ted and Future Ted.\" The article also says Ted is the one telling the story, so he may be an unreliable narrator."
BUSTLE = "The series finale \"Last Forever: Part Two\" settled the question, and the answer was \"Robin and the Blue French Horn.\" Ted's kids encouraged him to pursue Robin, and the article says his wife Tracy, \"The Mother,\" had been dead for six years by then."
WIKI_LOS = "The company was \"founded by Barbet Schroeder and \u00c9ric Rohmer in 1962\" and is headquartered in Paris. The infobox lists the same two founders. The page gives no employer or workplace for either founder."
WIKI_BS = "No employer or current workplace is named. The Career section says Les Films du Losange was founded by him at age 23 and produced French New Wave films. He \"resides in France\" and is married to actress Bulle Ogier."
LEFF = "Festival guest profile (2015 edition) lists him as a director and says he founded Les Films du Losange in 1963 at age 21. After years in the United States he returned to France, \"where he currently lives.\" It names no employer."
SWISS = "Interview published August 11, 2023 covering his career and his film \"Ricardo and Painting,\" shown at the Locarno Film Festival. It gives no current employer or workplace."
SMX = "Nationality field reads \"Americano.\" The bio says \"El cineasta, Scott Derrickson, naci\u00f3 el 16 de julio de 1966 en Denver, Colorado, Estados Unidos.\" The Nacimiento field gives 1977 (Denver, Colorado - Estados Unidos), which conflicts with the bio date."
SMS = "Nationality field reads \"Americano,\" and the breadcrumb says \"Guionista americano.\" The Nacimiento field gives 18 de marzo de 1977 with no birthplace. The bio opens \"Licenciado en Humanidades por la Biola University\"."
ENC = "The entry (International Dictionary of Films and Filmmakers, Gale) lists \"Nationality: American.\" Its birth line reads \"Born: Edward Davis Wood, Poughkeepsie, NY, 10 October 1924.\""
TMDB = "The English biography says Ed Wood \"was an American screenwriter, director, producer, actor, author, and editor.\" It gives his birth as October 10, 1924. The Hungarian translation places his birth in Poughkeepsie."

F = {
 "https://uk.news.yahoo.com/9-little-known-secrets-ted-130400701.html": YAHOO,
 "https://www.bustle.com/articles/19628-who-does-ted-end-up-with-on-how-i-met-your-mother-ugh-youre-not-going": BUSTLE,
 "https://www.Wikipedia.org/wiki/Les_Films_du_Losange": WIKI_LOS,
 "https://en.wikipedia.org/wiki/Barbet_Schroeder": WIKI_BS,
 "https://www.leffest.com/en/guests/barbet-schroeder": LEFF,
 "https://www.swissinfo.ch/eng/the-many-lives-of-barbet-schroeder--a-global-swiss-par-excellence/48726612": SWISS,
 "https://www.sensacine.com.mx/actores/actor-99042/biografia/": SMX,
 "https://www.sensacine.com/actores/actor-99042/biografia/": SMS,
 "https://encyclopedia.com/movies/dictionaries-thesauruses-pictures-and-press-releases/wood-edward-d-jr": ENC,
 "https://themoviedb.org/person/44763-edward-d-wood-jr/translations": TMDB,
}
ERR = {
 "https://www.tvinsider.com/swooon/gallery/how-i-met-your-mother-ending-ted-end-up-with/": "HTTP 403 Forbidden",
 "https://www.tvinsider.com/swooon/1177088/how-i-met-your-mother-ending-ted-end-up-with-robin-tracy-dead-finale/": "HTTP 403 Forbidden",
 "https://sportskeeda.com/pop-culture/how-i-met-your-mother-ending-explained-who-ted-end-with": "HTTP 405 Method Not Allowed",
 "https://www.aol.com/revisiting-controversial-met-mother-ending-160000000.html": "HTTP 404 Not Found",
 "https://collections.eastman.org/people/107879/barbet-schroeder/objects": "HTTP 403 Forbidden",
 "https://iffr.com/en/persons/barbet-schroeder": "HTTP 403 Forbidden",
 "https://www.entityfacts.com/en/person/scott-derrickson": "HTTP 403 Forbidden",
 "https://wiki.kidzsearch.com/wiki/Ed_Wood": "HTTP 403 Forbidden",
 "https://biola.atlassian.net/wiki/x/C4-p": "empty body (HTTP 202)",
 "https://stream4u.it.com/en/person/55499": "HTTP 404 Not Found",
}
# wikipedia mirrors under same article for Q9/10 uses same snippet
F["https://en.wikipedia.org/wiki/Barbet_Schroeder"] = WIKI_BS

def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

entries = [
 ("2hop__67660_81007", "full_question", None, "Who does the narrator on How I Met Your Mother end up with?", R1),
 ("2hop__67660_81007", "hop", 0, "Who narrates How I Met Your Mother?", R2),
 ("2hop__67660_81007", "hop_paraphrase", 0, "How I Met Your Mother narrator identity", R3),
 ("2hop__67660_81007", "hop", 1, "Who does Ted Mosby end up with?", R4),
 ("2hop__67660_81007", "hop_paraphrase", 1, "Ted Mosby's eventual wife", R5),
 ("sample_2hop_b", "full_question", None, "Where does the founder of Les Films Du Losange work at?", LOS),
 ("sample_2hop_b", "hop", 0, "Who founded Les Films Du Losange?", LOS),
 ("sample_2hop_b", "hop_paraphrase", 0, "Les Films Du Losange founder name", LOS),
 ("sample_2hop_b", "hop", 1, "Where does Barbet Schroeder work?", S9),
 ("sample_2hop_b", "hop_paraphrase", 1, "Barbet Schroeder current workplace", S10),
 ("sample_3hop_c", "full_question", None, "Were Scott Derrickson and Ed Wood of the same nationality?", D11),
 ("sample_3hop_c", "hop", 0, "What nationality is Scott Derrickson?", D12),
 ("sample_3hop_c", "hop_paraphrase", 0, "Scott Derrickson nationality director", D13),
 ("sample_3hop_c", "hop", 1, "What nationality was Ed Wood?", D14),
 ("sample_3hop_c", "hop_paraphrase", 1, "Ed Wood director nationality", D15),
 ("sample_3hop_c", "hop", 2, "Scott Derrickson Ed Wood same nationality comparison", D16),
 ("sample_3hop_c", "hop_paraphrase", 2, "compare nationalities Scott Derrickson Ed Wood", D17),
]

data = json.load(open(OUT, encoding="utf-8"))
have = {(e["qid"], e["search_type"], e["hop_idx"]) for e in data}
for qid, st, hi, q, res in entries:
    if (qid, st, hi) in have:
        continue
    out = []
    for i, (title, url) in enumerate(res):
        r = {"title": title, "url": url, "snippet": ""}
        if i >= 3:
            r["fetch_error"] = "not_in_top_3"
        elif any(j in url for j in JUNK):
            r["fetch_error"] = "skipped_low_value_domain"
        elif url.lower().endswith(".pdf"):
            r["fetch_error"] = "pdf_not_fetched"
        elif url in F:
            r["snippet"] = F[url]
        else:
            r["fetch_error"] = ERR.get(url, "fetch failed")
        out.append(r)
    data.append({"qid": qid, "search_type": st, "hop_idx": hi, "query_text": q, "results": out, "timestamp": now()})
    json.dump(data, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(len(data))
