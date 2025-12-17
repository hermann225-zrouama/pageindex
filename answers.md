# Résultats du Questionnaire

**Date:** 2025-12-17 12:23:11

**Nombre de questions:** 10

---

## Question 1

**Query:** Mon client a appelé le 2/12/24, il me dit qu'il est arrivé le 01/12/24 à index 136. Quand je consulte les AT, le branchement a été fermé le 21/11/24 avec un index de 136, et réouvert le 03/12/24. à quelle date dois-je l'abonner ?

### Réponse

D'après le contexte fourni, votre client a signalé une arrivée le **01/12/2024** avec un index de **136**, tandis que les **AT (Accès Techniques)** indiquent :
- **Fermeture du branchement** : **21/11/2024** (index 136)
- **Réouverture du branchement** : **03/12/2024**

### **Date d'abonnement à retenir**
1. **Principe général** :
   - L'abonnement doit couvrir la **période effective de consommation**, c'est-à-dire à partir de la **date de réouverture du branchement** (sauf si le client prouve une occupation antérieure).
   - **L'index 136** (identique à la fermeture) suggère qu'**aucune consommation n'a eu lieu entre le 21/11 et le 03/12**.

2. **Analyse des dates** :
   - **21/11/2024** : Fermeture (index 136) → **Pas de consommation possible**.
   - **01/12/2024** : Date déclarée par le client (arrivée). **À vérifier** :
     - Si le client était **physiquement présent** avant le 03/12, une **régularisation** peut être nécessaire (ex. : facturation à partir du 01/12).
     - Si le client **n'a pas consommé** (pas d'usage d'eau), la date de réouverture (**03/12/2024**) prime.
   - **03/12/2024** : Réouverture → **Date officielle de début d'abonnement** (sauf preuve contraire).

3. **Recommandation** :
   - **Abonner à partir du 03/12/2024** (date de réouverture du branchement).
   - **Si le client insiste sur le 01/12** :
     - Vérifier s'il a **effectivement consommé** (ex. : relevé d'index intermédiaire, témoignage).
     - En l'absence de preuve, **maintenir le 03/12** pour éviter une facturation sans consommation.

4. **Cas particulier (mensualisation)** :
   - Si le client adhère à la **mensualisation**, appliquer les règles du document **[PNATCLI11]** :
     - **Échéancier** calculé sur **10 mois** (pour éviter les doublons avec les factures semestrielles).
     - **Montant estimé** basé sur la composition du foyer (ex. : 15 €/

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Unknown

---

## Question 2

**Query:** Mon client a appelé le 27/06, il est arrivé le 01/05 . Pas d'index. Je consulte les AT : branchement fermé le 27/06 , et réouvert le 28/06 IND 646. Le prédécesseur est résilié le 04/06 IND 627 sans payer ses dernières factures.L'état des lieux fourni ne précise pas l'index. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni :**
Votre client a emménagé le **01/05**, mais son abonnement n’a pas encore été activé. Les **AT (Appels Techniques)** indiquent :
- Un **branchement fermé le 27/06** (suite à la résiliation du prédécesseur).
- Une **réouverture le 28/06** avec un **index de 646**.
- Le **prédécesseur résilié le 04/06** avec un **index de 627**, sans régler ses dernières factures.
- **Aucun index** n’est mentionné sur l’état des lieux fourni.

---

### **Date et index à utiliser pour l'abonnement**
#### **1. Règle générale (sans justificatif d'index)**
- **Date d'abonnement** : **Date de résiliation du prédécesseur** (soit le **04/06**).
- **Index à utiliser** : **Index de résiliation du prédécesseur** (soit **627**).
  - *Justification* : En l’absence d’index sur l’état des lieux, la procédure impose de se baser sur l’index de résiliation de l’ancien occupant (cf. **Source : [20251106 Rappel métier écart consommation entre 2 abonnés v5 | Traitement en back office]**).

#### **2. Vérification de l'écart de consommation**
- **Écart entre index** :
  - Index de réouverture (28/06) : **646**
  - Index de résiliation (04/06) : **627**
  - **Écart = 19 m³** (supérieur à la tolérance de **5 m³**).
- **Action requise** :
  - **Ouvrir un dossier ARR avec intervention** pour vérifier l’écart.
  - **Préciser dans le commentaire** : *"Écart de consommation > 5 m³ entre l’index de résiliation (627) et l’index de réouverture (646). Vérification nécessaire."*
  - *Source : [20251106 Rappel métier écart consommation entre 2 abonnés v5 | Cas d’une demande d’arrivée avec remise en eau]*.

#### **3. Exception (si justificatif fourni ultérieurement)**
- Si le client transmet **ultérieurement un état des lieux avec index** :
  - **Abonnement rétroactif** à la date et à l’index indiqués (dans la limite de **2 ans** de prescription).
  - *Source : [20251106 Rappel métier écart consommation entre 

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20251106 Rappel métier écart consommation entre 2 abonnés v5

---

## Question 3

**Query:** Mon client a appelé le 04/07/25, il me dit qu'il souhaite reprendre le contrat de sa mère décédée. Il nous fourni l'index du jour et un acte d'hérédité. a quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni**, votre client souhaite reprendre le contrat de sa mère décédée en tant qu'héritier. Voici la procédure à suivre et la date d'abonnement applicable :

---

### **1. Étapes à suivre**
**Ne pas modifier directement le titulaire du contrat** (ni le RIB) sur la base des documents fournis (acte d'hérédité + index).
**Actions requises :**
1. **Résilier le contrat au nom du prédécesseur (la mère)** :
   - **Sans frais** (conformément à la procédure de succession).
   - **Date de résiliation** : **04/07/2025** (date de l'appel du client).
     *Justification* : La résiliation prend effet à la date de la demande, sauf mention contraire dans les CGV.

2. **Créer un nouveau contrat au nom de l'héritier** :
   - **Date d'effet** : **04/07/2025** (même date que la résiliation pour éviter une interruption de service).
   - **Index** : Utiliser l'**index du jour** fourni par le client.
   - **Documents à joindre** :
     - Acte d'hérédité (pour justifier la reprise du contrat).
     - Pièce d'identité du nouveau titulaire.

3. **Transmettre le dossier à l'équipe dédiée** :
   - La **mutation officielle** (changement de titulaire) sera traitée en interne par le service succession.
   - **Ne pas modifier le RIB** avant validation par ce service.

---

### **2. Points de vigilance**
- **Pas de double facturation** : Vérifier que la résiliation et la création du nouveau contrat sont bien synchronisées.
- **Communication au client** :
  - L'informer que le contrat sera **temporairement sous son nom** après le 04/07/2025, mais que la mutation définitive dépendra du traitement interne.
  - Préciser que **aucun frais de résiliation** ne lui sera facturé.

---

### **3. Délai de traitement**
- **Résiliation** : Effective sous **48h** (à confirmer selon les CGV).
- **Nouveau contrat** : Activation sous **24 à 48h** après création.
- **Mutation définitive** : Délai variable (généralement **10 à 15 jours** après réception des documents complets par le service succession).

---

**Source(s):**
- [Rappel métier - Succession et modification titulaire v4 | Section "Important" et "Principe"](20250610- Rappel métier - Succession et modification titulaire v4)
- Procédure interne SUEZ (version 

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20250610- Rappel métier - Succession et modification titulaire v4

---

## Question 4

**Query:** Mon client a appelé le 14/08/24, il me dit qu'il est arrivé le 17/05/24. Pas d'index. Quand je consulte les AT, le branchement a été fermé le 30/07/24, et réouvert le 14/08/24, avec le même index. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni, voici la marche à suivre pour abonner votre client dans le cadre d'une mutation rétroactive avec un historique de fermeture/réouverture du branchement :**

---

### **1. Synthèse de la situation**
- **Date d'arrivée du client** : **17/05/2024** (à prendre en compte pour l'abonnement).
- **Historique du branchement** :
  - Fermeture le **30/07/2024**.
  - Réouverture le **14/08/2024** avec le **même index** (pas de consommation enregistrée entre les deux dates).
- **Appel du client** : **14/08/2024** (date de réouverture = jour J).

---

### **2. Règles à appliquer**
#### **Cas général (mutation rétroactive)**
- **L'abonnement doit être rétroactif** à la date d'arrivée du client (**17/05/2024**), **même si le contrat du prédécesseur était actif jusqu'au 30/07/2024**.
  - **Exception** : Si le client s'oppose à une dette du prédécesseur (dans la limite de **2 ans de prescription**), l'abonnement reste possible.
- **Pas d'index disponible** : Utilisez la date d'arrivée comme point de départ pour le calcul des consommations (à ajuster ultérieurement si un index est fourni).

#### **Spécificités de ce dossier**
- **Fermeture/réouverture sans changement d'index** :
  - Le branchement a été **fermé administrativement** (ex. : résiliation du prédécesseur) puis **réactivé** pour le nouveau client.
  - **Aucune consommation n'a été enregistrée** entre le 30/07 et le 14/08 (même index), donc **pas d'impact sur la facturation**.
- **Date d'abonnement** :
  - **Priorité à la date d'arrivée** (17/05/2024) pour couvrir la période où le client occupait les lieux.
  - **La réouverture du 14/08/2024** est une **conséquence administrative** (pas une date d'emménagement).

---

### **3. Étapes à suivre**
1. **Créer l'abonnement rétroactif** :
   - **Date de début** : **17/05/2024** (date d'arrivée du client).
   - **Index de départ** : À laisser vide ou à 0 (à mettre à jour ultérieurement si le client fournit un index).
   - **

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20251106 Rappel métier mutation rétroactive v2

---

## Question 5

**Query:** Mon client a appelé le 7/08/25, il me dit qu'il est arrivé le 07/06/22. Pas d'index. Quand je consulte les AT, le branchement a été fermé le 7/08/25, et réouvert le 8/08/25. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni, voici la démarche à suivre pour déterminer la date d'abonnement du client :**

### **1. Analyse de la situation**
- **Date d'arrivée du client** : **07/06/2022** (déclarée par le client).
- **Appel du client** : **07/08/2025** (date de contact).
- **Historique du branchement** :
  - **Fermeture** : **07/08/2025** (jour de l'appel).
  - **Réouverture** : **08/08/2025** (lendemain).

**Problème** : **Aucun index de consommation** n'est disponible pour calculer une facture précise.

---

### **2. Règles d'abonnement applicables**
D'après le document **[PNATCLI11 - Mensualisation et prélèvement automatique]** :
- **Pour un nouveau client sans historique de consommation** :
  - L'abonnement doit être **estimé** en fonction de la composition du foyer (ex. : 1 pers = 15 €/mois, 2 pers = 20 €/mois).
  - **La date d'effet de l'abonnement** correspond à la **date de réouverture du branchement** (sauf si le client prouve une occupation antérieure).

---

### **3. Date d'abonnement à retenir**
- **Cas standard** :
  - **Date de réouverture du branchement** = **08/08/2025** (date à privilégier).
  - *Justification* : Le branchement était **fermé le 07/08/2025** (jour de l'appel), donc l'abonnement ne peut pas démarrer avant sa réouverture.

- **Exception possible** :
  - Si le client **fournit une preuve d'occupation** (ex. : bail, facture d'électricité) **avant le 07/08/2025**, la date d'abonnement peut être **antidatée** (ex. : 07/06/2022).
  - *À vérifier* : **Aucune preuve n'est mentionnée dans votre contexte**.

---

### **4. Étapes à suivre**
1. **Vérifier la présence d'un historique** :
   - Consulter le **SI Clientèle** pour confirmer l'absence d'index ou de factures antérieures.
2. **Proposer une estimation** :
   - Baser l'échéancier sur la **composition du foyer** (ex. : 15 €/mois pour 1 personne).
3. **Informer le client** :
   - Lui expliquer que l'abon

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Unknown

---

## Question 6

**Query:** Mon client a appelé le 19/06/25, il me dit qu'il est arrivé le 18/06/24 à index 0. Quand je consulte le compte, je vois que le branchement est fermé depuis la pose du compteur et réouvert le 20/06/25. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni, voici la marche à suivre pour abonner votre client dans le cadre d'une mutation rétroactive avec un compteur fermé puis réouvert :**

---

### **1. Analyse de la situation**
- **Date d'arrivée déclarée par le client** : **18/06/2024** (index 0).
- **État du compteur** :
  - Fermé depuis la pose (date non précisée).
  - **Réouvert le 20/06/2025**.
- **Appel du client** : **19/06/2025** (soit **1 jour avant la réouverture**).

**Points clés** :
- Le client demande un abonnement **rétroactif** (date antérieure au jour J = 19/06/2025).
- Le compteur était **inactif** entre sa pose et le 20/06/2025.

---

### **2. Règles à appliquer**
D'après le **rappel métier mutation rétroactive v2** (06/11/2025) :
- **L'abonnement doit être réalisé aux dates communiquées par le client** (ici, **18/06/2024**), **même si** :
  - Le compteur était fermé.
  - L'ancien occupant avait une dette (sous réserve de prescription de 2 ans).
- **Exception** : Si la date demandée est **antérieure au jour J** (19/06/2025), le conseiller doit :
  1. **Annuler toute intervention technique en cours** (ex : réouverture du compteur).
  2. **Ne pas traiter la demande en front office** → **Transférer en back office** pour validation.

---

### **3. Étapes à suivre**
1. **Vérifier la cohérence des dates** :
   - Le client déclare être arrivé le **18/06/2024**, mais le compteur n'a été réouvert que le **20/06/2025**.
   - **Risque** : Incohérence entre la date d'arrivée et l'état du compteur (fermé pendant 1 an).

2. **Demander des justificatifs** (si nécessaire) :
   - Preuve de la date d'emménagement (bail, facture d'électricité précédente, etc.).
   - **Objectif** : Confirmer que le client était bien présent avant la réouverture du compteur.

3. **Traiter la demande** :
   - **Si la date du 18/06/2024 est validée** :
     - **Abonner rétroactivement à cette

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20251106 Rappel métier mutation rétroactive v2

---

## Question 7

**Query:** Mon client a appelé le 23/07/25, il me dit qu'il est arrivé le 21/06/25 à index 1474. Quand je consulte les AT, le branchement a été fermé le 23/07/25, et réouvert le même jour à index 1494. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni, voici la démarche à suivre pour déterminer la date d'abonnement du client :**

### **1. Analyse des éléments clés**
- **Date d'arrivée du client** : **21/06/2025** (index 1474).
- **Fermeture du branchement** : **23/07/2025** (jour de l'appel).
- **Réouverture du branchement** : **23/07/2025** (même jour, index 1494).
- **Écart d'index** : **20 m³** (1494 - 1474) consommés entre le 21/06 et le 23/07.

---

### **2. Règles de facturation applicables**
- **Principe** : L'abonnement doit couvrir la **période réelle d'occupation** du logement, même si le branchement a été temporairement fermé.
- **Cas spécifique** :
  - Si le branchement a été **fermé et réouvert le même jour**, cela suggère une **intervention technique** (ex. : vérification, mise à jour) plutôt qu'une vacance du logement.
  - Le client était **effectivement présent** depuis le **21/06/2025** (date d'arrivée déclarée).

---

### **3. Date d'abonnement à retenir**
- **Date de début d'abonnement** : **21/06/2025** (date d'arrivée du client).
  - **Justification** :
    - La fermeture/réouverture le 23/07 est un **événement ponctuel** sans impact sur la continuité de l'occupation.
    - Les **20 m³ consommés** confirment une utilisation du service entre le 21/06 et le 23/07.
- **Période à facturer** :
  - Du **21/06/2025** au **23/07/2025** (consommation réelle).
  - Puis **abonnements mensuels** à partir du 23/07/2025 (selon l'échéancier type, ex. : prélèvements le 30 de chaque mois).

---

### **4. Actions à mener**
1. **Créer l'abonnement** avec une **date de début au 21/06/2025**.
2. **Facturer la consommation** :
   - **20 m³** pour la période du 21/06 au 23/07/2025.
   - **Abonnement** proratisé sur cette période

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Unknown

---

## Question 8

**Query:** Mon client a appelé le 20/10/25, il me dit qu'il est arrivé le 15/01/25 à index de 1356. Le branchement est ouvert. Le prédécésseur a résilié son contrat le 30/05/25 avec index de 1422 et a réglé sa facture de solde. à quelle date dois-je l'abonner ?

### Réponse

**D'après le contexte fourni, voici la marche à suivre pour abonner votre client successeur :**

### **1. Contexte clé**
- **Prédécesseur** :
  - Résiliation le **30/05/25** (index final : **1422**).
  - Facture de solde réglée → **contrat clôturé à cette date**.
- **Nouveau client** :
  - Arrivé le **15/01/25** (index initial : **1356**).
  - **Branchement ouvert** (compteur accessible).

---

### **2. Date d'abonnement à appliquer**
**Règle générale** :
- Le contrat du successeur doit débuter **à la date de son emménagement effectif** (remise des clés/état des lieux), **sous réserve que le prédécesseur ait bien résilié avant**.

**Application à votre cas** :
- **Date d'arrivée du successeur** : **15/01/25** (avant la résiliation du prédécesseur le 30/05/25).
  - **Problème** : Le prédécesseur était encore titulaire du contrat jusqu'au 30/05/25. Le successeur ne peut pas être abonné sur la même période.
  - **Solution** :
    - **Abonnement rétroactif** à la **date de résiliation du prédécesseur** (30/05/25), **sauf si** :
      - Le successeur prouve qu'il a occupé les lieux avant (ex. : bail, état des lieux).
      - Dans ce cas, **ouvrir un dossier de régularisation** pour couvrir la période du **15/01/25 au 30/05/25** (consommations à facturer au prédécesseur ou au successeur selon accord).

**Recommandation** :
- **Vérifier les preuves d'occupation** (bail, état des lieux) pour confirmer la date du 15/01/25.
- **Si pas de preuve** : Abonner à partir du **30/05/25** (date de résiliation du prédécesseur).
- **Si preuve valide** : Abonner à partir du **15/01/25** et régulariser la période de chevauchement avec le prédécesseur.

---

### **3. Étapes à suivre**
1. **Créer le client successeur** en tant que prospect (si ce n'est pas déjà fait).
2. **Ouvrir un dossier "Arrivée Client signalée ASI"** (qualifié "information par le prédécesseur").
3. **Vérifier l'index** :
   - Index au 15

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 9

**Query:** Client nous contacte le 25/10/25 pour résilier son contrat et nous fournis un index du jour de 78. Le client à une dette de 550€ et un échéancier de paiement mensuel allant jusqu'en mars 2026. que dois je faire ?

### Réponse

**D'après le contexte fourni**, un client souhaite résilier son contrat le **25/10/25** tout en ayant :
- Un **solde débiteur de 550€**
- Un **échéancier de paiement en cours** (jusqu'en mars 2026)
- Un **index de consommation** fourni (78).

Voici la **procédure à suivre** :

---

### ### 1. **Informer le client des conséquences de la résiliation**
- **Résiliation du prélèvement automatique** :
  - Le client doit **régler le solde débiteur (550€) sous 15 jours** (délai impératif).
  - **Aucun échéancier ne peut être maintenu** après résiliation (cf. procédure PNATCLI11).
  - **Préciser** que le prélèvement sera désactivé **après règlement intégral** du solde.

- **Résiliation de la mensualisation** (si applicable) :
  - Le solde restant dû sera **réclamé sur la prochaine facture** (pas de délai de 15 jours, mais paiement immédiat attendu).
  - **Aucun achèvement de dette** ne peut être accordé.

---

### ### 2. **Étapes concrètes à suivre**
1. **Vérifier le type de résiliation demandée** :
   - Prélèvement automatique **et/ou** mensualisation ?
   - Adapter la réponse en fonction (cf. sections [Résiliation] et [Prélèvement Automatique] de PNATCLI11).

2. **Enregistrer l’index fourni (78)** :
   - S’assurer qu’il est bien pris en compte pour la **facture de clôture**.

3. **Notifier le client par écrit** (email/lettre) :
   - **Montant dû** : 550€.
   - **Délai de paiement** : 15 jours (si résiliation du prélèvement) **ou** à la prochaine facture (si résiliation de la mensualisation).
   - **Moyens de paiement** : Préciser les options (CB, chèque, virement, etc.).
   - **Conséquences en cas de non-paiement** : Mise en recouvrement, frais supplémentaires.

4. **Annuler l’échéancier en cours** :
   - **Ne pas maintenir les prélèvements** après la résiliation.
   - **Clôturer l’échéancier** dans le système (date de fin : 25/10/25).

5. **Proposer une aide en cas de difficultés** :
   - Si le client évoque des problèmes financiers, l’orienter vers la procédure **P_NAT

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- PNATCLI11 Mensualisation et prélèvement automatique

---

## Question 10

**Query:** Le client appelle le 21/10/25 et souhaite résilier son contrat à la date du 23/09/2025 et nous envoie un état des lieux sortant ainsi qu’une photo du compteur au 23/09/25. Que dois-je faire ?

### Réponse

**D'après le contexte fourni**, le client souhaite résilier son contrat **à une date antérieure à son appel (23/09/25)** et fournit des justificatifs (état des lieux sortant + photo du compteur). Voici la procédure à suivre :

---

### **1. Vérifier la recevabilité de la demande**
- **Date de résiliation demandée (23/09/25)** :
  - **Problème** : La résiliation ne peut pas être rétroactive. Le client reste redevable des consommations et abonnements **jusqu’à la date de sa demande effective** (21/10/25), sauf si :
    - Il a **signalé son départ par écrit avant le 23/09/25** (ex. : courrier, email).
    - La relève du compteur a été effectuée **par un technicien** à la date du 23/09/25 (fiche G2 remplie).
  - **À faire** :
    1. **Confirmer** si le client a déjà signalé son départ **avant le 23/09/25** (vérifier l’historique des contacts).
    2. **Vérifier** si une **intervention technique (AT)** a été réalisée le 23/09/25 (fiche G2) :
       - Si **oui** : La date de résiliation peut être fixée au 23/09/25.
       - Si **non** : La résiliation prendra effet **à la date de la demande (21/10/25)**.

- **Justificatifs fournis** :
  - L’**état des lieux sortant** et la **photo du compteur** n’ont **pas de valeur juridique** pour une résiliation rétroactive (cf. procédure).
  - **Exception** : Si la photo montre une **relevé de compteur lisible et daté**, elle peut servir de preuve pour une **régularisation ultérieure** (voir point 3).

---

### **2. Traiter la résiliation**
#### **Cas 1 : Résiliation possible au 23/09/25**
- **Conditions remplies** :
  - Preuve écrite du client **avant le 23/09/25** **OU** intervention technique (fiche G2) le 23/09/25.
- **Actions** :
  1. **Clôturer le contrat** à la date du 23/09/25.
  2. **Ouvrir un Dossier ASI** (Arrêt de Service Immédiat) avec la branche **"Logement vacant"**.
 

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

