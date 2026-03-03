**VA Proposal** (migliorare/valutare l'utilità di una visualizzazione)

**Focus**: distanza euclidea (PCA vs MDS euclideo)

Tre step principali:

1. Riportare la qualità di PCA (cosa non va; quali sono i falsi positivi) e MDS euclideo (sia falsi positivi che negativi, ...) rispetto a un dataset => wine (già visto a lezione). Questa voce deve prevedere una visualizzazione doppia (e sincronizzata) dei due "scatterplot" con: **scala di colori** (utile a rappresentare l'indice di falsità positiva o negativa dei punti), **metriche di qualità** (ad esempio uno score globale raffigurante la somma dei punti coinvolti, ...), e un **primo giudizio sul dataset** (i.e., l'80% dei punti coincide, ...). Da notare che, per i falsi positivi e negativi (blu/rossi, bisogna attribuire loro una **definizione** appropriata, e impiegare metriche che non siano semplicemente di on/off (considerare il loro peso, ossia quanto falso positivo o negativo); o la prima lettera del nome.

2. Introdurre delle label (astratte, come i produttori di vino, e dipendenti dai dati). L'obiettivo è valutare le discrepanze tra 3 cluster astratti (che non dipendono pertanto dal calcolo della distanza euclidea, e che quindi risultano indipendenti dai valori del dataset) e 3 inerenti ai dati, utilizzando tecniche come la k-mean. Questa visualizzazione deve offrire la possibilità di mostrare le porzioni dei cluster che sembrerebbero non appartenere a quello di loro pertinenza (tramite centroidi => qualche strumento/reference dovrebbe essere già stata pubblicata da Santucci stesso). In aggiunta, devono essere presenti delle metriche di qualità e filtri (per la rimozione di falsi positivi o negativi), utilizzando sia MDS euclideo che PCA.

3. Evidenziare quanto la proiezione bidimensionale ottenuta è fedele ai cluster/spazi multidimensionali originali. La richiesta è quella di realizzare uno switch che, tramite l'impiego di k-mean, possa permettere di visualizzare le possibili intersezioni/aree di interesse tra gli elementi menzionati (3 cluster originali, dei "produttori" ...).

---

**Goal**: **Fedeltà di PCA** (valutazione e giudizio) w.r.t. distanza euclidea e dimostrazione di quanto MDS euclideo sia migliore => eventualmente migliorando la qualità di quello che si sta osservando.