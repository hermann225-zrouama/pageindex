# Résultats du Questionnaire

**Date:** 2025-12-18 17:32:37

**Nombre de questions:** 13

---

## Question 1

**Query:** Mme YYYY envoie le message suivant : « Bonjour, Je vous informe avoir déménagé le 01 août 2025 et vous transmets les documents attestant  que je n’occupe plus le logement. Je vous remercie de bien vouloir procéder à la résiliation de mon contrat et de ne plus me  facturer après cette date. Cordialement, Mme YYYY Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni, Mme YYYY demande la résiliation de son contrat suite à son déménagement le 01/08/2025, avec transmission des justificatifs de départ.**

### **Étapes à suivre pour traiter la demande**

#### **1. Vérification et mise à jour des données client**
- Confirmer l’**email** (MAIL123@MAIL) et le **numéro de téléphone** dans le système.
- **Source :** [Procédure Départ Client | DETAIL DES OPERATIONS MENEES]

#### **2. Ouverture du dossier de résiliation**
- Créer un **Dossier "Départ client Signalé - DSI"** dans le système.
- Renseigner la **date prévisionnelle de départ** : **01/08/2025** (date effective de sortie).
- **Source :** [Procédure Départ Client | DETAIL DES OPERATIONS MENEES]

#### **3. Traitement de la résiliation**
- **Date de fin de contrat** = **01/08/2025** (date de sortie des lieux, justifiée par les documents transmis).
- Le client reste redevable des **consommations et abonnements jusqu’à cette date**.
- **Source :** [Procédure Départ Client | TRAITER LES DEMANDES DE RESILIATION D'ABONNEMENT]

#### **4. Ouverture d’un Dossier ASI (si logement vacant)**
- Si le logement est **désormais inoccupé**, ouvrir un **Dossier ASI** avec la branche **"Logement vacant"**.
- **Source :** [Procédure Départ Client | TRAITER LES DEMANDES DE RESILIATION D'ABONNEMENT]

#### **5. Facturation finale**
- Une **facture d’arrêt de compte** sera émise pour régulariser les consommations jusqu’au 01/08/2025.
- **Source :** [Procédure Départ Client | TRAITER LES DEMANDES DE RESILIATION D'ABONNEMENT]

### **Prochaines actions**
- **Accuser réception** du mail de Mme YYYY en confirmant :
  - La prise en compte de sa demande.
  - La date de résiliation (01/08/2025).
  - L’envoi prochain de la facture de clôture.

**Source(s) :**
- [Procédure Départ Client | TRAITER LES DEMANDES DE RESILIATION D'ABONNEMENT]
- [Procédure Départ Client | DETAIL DES OPERATIONS MENEES]

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 2

**Query:** Mme YYYY envoie le message suivant : « Bonjour, J’ai souscrit un contrat d’eau à une nouvelle adresse, toutefois le contrat n’est pas encore  visible, alors que le mandat SEPA a bien été signé. Je vous remercie de bien vouloir vérifier la prise en compte de mon emménagement. Cordialement, Mme YYYY Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni**, Mme YYYY signale que son **contrat d’eau n’est pas encore visible** malgré la signature du mandat SEPA pour son nouvel emménagement. Voici la démarche à suivre pour traiter sa demande :

---

### **1. Vérification immédiate du dossier**
**Actions à réaliser :**
1. **Accéder à TPMC** (Tableau de Pilotage Multicanal Client) et/ou à la **base des services** via le compte client **Odyssée** (écran "vue 360°").
2. **Rechercher l’adresse d’emménagement** de Mme YYYY :
   - Via le **site** ou le **matricule du compteur** (si fourni).
   - Vérifier si le **contrat est en cours de création** ou en attente de validation.
3. **Contrôler le statut du mandat SEPA** :
   - Confirmer que le mandat est **bien enregistré** et **validé** (délai de traitement possible : 24 à 48h).
   - Si le mandat est en erreur, **relancer le processus** ou informer Mme YYYY des étapes restantes.

---

### **2. Points de contrôle spécifiques**
**À vérifier systématiquement :**
- **Cohérence des index** :
  - Si l’index transmis par Mme YYYY **ne correspond pas** à celui du prédécesseur, une **intervention de relève** sera déclenchée (à notifier au client).
- **Type de logement** :
  - Confirmer s’il s’agit d’une **maison** ou d’un **appartement** (pour éviter les doublons ou erreurs d’adresse).
- **Délais standards** :
  - **Visibilité du contrat** : 24 à 72h après signature du mandat SEPA (selon la procédure interne).
  - **Activation du service** : Sous 5 jours ouvrés après validation complète.

---

### **3. Réponse à apporter à Mme YYYY**
**Exemple de message personnalisé :**
> *"Bonjour Mme YYYY,
> Nous avons bien pris en compte votre demande concernant votre emménagement à [adresse précise]. Après vérification :
> - Votre **mandat SEPA est bien enregistré** et en cours de traitement.
> - Votre **contrat sera visible sous 24 à 48h** dans votre espace client [lien vers l’espace].
> Nous restons à votre disposition pour toute question complémentaire. Cordialement, [Votre nom/prénom] – Service Client [Nom du centre]."*

**Si anomalie détectée :**
> *"Nous constatons un délai anormal dans le traitement de votre dossier. Nous transmettons votre demande à notre service technique pour résolution sous 24h. Nous vous tiendrons informée par retour de mail."*

---

### **4. Escalade si nécessaire**
**Cas nécessitant une escalade :**
- **Mandat SEPA non visible** après 48h → Contacter le service **Back-Office** (réf. P_NAT_CLI_12, section 6.2).
- **Index incohérent** → Déclencher une **relève d’index** via le formulaire dédié (lien interne).
- **Adresse introuvable** → Vérifier avec le client la **précision de l’adresse** (étage, bâtiment, etc.).

---

**Source(s):**
- **Procédure nationale Arrivée Client V8** | Sections :
  - *6.2 Traitement de l’arrivée client le jour de son emménagement* (p.6).
  - *Aide au discours arrivée client* (identification des informations).
- **Réf. P_NAT_CLI_12** (indice 8, mise à jour 08/06/2022).

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procedure nationale Arrivée Client V8

---

## Question 3

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Je souhaite procéder à la résiliation de mon contrat d’eau suite à un déménagement. N’ayant pas réussi à effectuer la démarche en ligne ni par téléphone, je vous contacte par ce  message. Merci de me recontacter afin de finaliser la résiliation. Cordialement, M XXXXXX Mail : MAIL123@MAIL Tél : 00123456789 »

### Réponse

**D'après le contexte fourni**, M. XXXXXX souhaite résilier son contrat d’eau dans le cadre d’un déménagement et demande un rappel pour finaliser la démarche. Voici la procédure à suivre pour traiter sa demande, conformément aux directives du centre de relation client.

---

### **1. Étapes immédiates à engager**
1. **Contacter le client sous 24h** (par téléphone ou email) pour :
   - Confirmer la réception de sa demande.
   - Lui proposer une assistance personnalisée pour finaliser la résiliation.

2. **Vérifier les tentatives précédentes** :
   - Demander au client les détails des échecs (ex. : problème technique sur le site, indisponibilité du service téléphonique).
   - **Documenter ces éléments** dans le dossier client pour analyse ultérieure.

---

### **2. Procédure de résiliation à appliquer**
#### **A. Promotion du compte en ligne (principe prioritaire)**
- **Rappeler au client** qu’il peut effectuer la résiliation **directement via son compte en ligne** :
  - Site : [https://www.toutsurmoneau.fr](https://www.toutsurmoneau.fr)
  - Avantages : **démarche autonome, rapide et sécurisée**.
- **Proposer un accompagnement** (ex. : envoi d’un guide pas-à-pas par email ou partage d’écran si contact téléphonique).

#### **B. Ouverture du dossier "Départ client"**
1. **Créer un dossier "Départ client Signalé - DSI"** dans le système de gestion.
2. **Renseigner les informations obligatoires** :
   - **Date prévisionnelle de départ** (à demander au client).
   - Coordonnées du client (email : MAIL123@MAIL, tél : 00123456789).
   - **Motif de résiliation** : déménagement (à préciser si possible : nouvelle adresse, date d’emménagement).

#### **C. Relevé de compte et facturation**
- **Proposer un "relevé confiance"** (si applicable) pour éviter les litiges sur la consommation finale.
- **Informer le client** des éventuels frais de résiliation (ex. : **frais de fermeture de compte**) prévus par le règlement de service.
  - **Préciser le montant et la date de prélèvement** si applicable.

#### **D. Confirmation écrite**
- **Envoyer un email de confirmation** au client avec :
  - Récapitulatif de la date de résiliation effective.
  - Instructions pour le relevé final (si nécessaire).
  - Coordonnées du service client en cas de question.

---

### **3. Points de vigilance**
- **Délai de traitement** : La résiliation doit être effective **à la date demandée par le client** (sous réserve de la transmission des informations requises).
- **Preuve de déménagement** : Si le règlement de service l’exige, demander une **justificatif** (ex. : bail, attestation de déménagement).
- **Suivi post-résiliation** :
  - Vérifier que le client reçoit bien sa **facture de clôture**.
  - Proposer un **transfert de contrat** si le client reste dans la zone de desserte du service.

---

### **4. Canaux de contact à privilégier**
- **Téléphone** : Pour un échange immédiat et personnalisé (idéal pour les urgences).
- **Email** : Pour les confirmations écrites et les pièces jointes (ex. : guide de résiliation).
- **Chat en ligne** : Si le client préfère une assistance textuelle en temps réel.

---

**Source(s):**
- Procédure Nationale "Départ Client" | Section **TRAITER LES DEMANDES DE RESILIATION D'ABONNEMENT** (Réf : P_NAT_CLI_13, Indice 8, Page 12).
- Principes clés : **Site internet et compte en ligne**, **Relevé confiance**, **Date de résiliation**.

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 2
- **Contexte:** ~1018 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 4

**Query:** me YYYY envoie le message suivant : « Bonjour, Je vous transmets mon état des lieux de sortie accompagné de la photo du compteur. Ces éléments sont envoyés afin de permettre la résiliation de mon contrat. Cordialement, Mme YYYY Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni, Mme YYYY souhaite résilier son contrat en transmettant son état des lieux de sortie et la photo du compteur.**

### **Étapes à suivre pour traiter la demande de résiliation**

1. **Vérification des éléments reçus**
   - Confirmer la réception de :
     - L’**état des lieux de sortie**
     - La **photo du compteur** (vérifier sa lisibilité et la présence des index)
   - **Si un élément manque** : Demander à Mme YYYY de le renvoyer.

2. **Mise à jour des données client**
   - Vérifier et actualiser si nécessaire :
     - **Email** (MAIL123@MAIL)
     - **Téléphone** (non fourni dans le message → à demander si absent du dossier)

3. **Rappel des obligations contractuelles**
   - Informer Mme YYYY que :
     - **Le contrat reste actif** jusqu’à la résiliation effective (accès au compteur validé).
     - **Elle reste redevable** des consommations et abonnements jusqu’à la clôture du contrat.

4. **Vérification de l’accès au compteur**
   - **Si le compteur est accessible** (photo valide) :
     - Finaliser la résiliation après validation des index.
   - **Si le compteur n’est pas accessible** (ex. : absence de photo ou compteur non visible) :
     - Proposer un **rendez-vous** pour relevé ou demander une **nouvelle photo conforme**.
     - **Préciser** : Sans accès, le contrat reste à son nom et les factures continuent.

5. **Clôture du contrat d’assurance (si applicable)**
   - Vérifier si Mme YYYY a un contrat d’assurance lié (ex. : Homeserve).
   - **Si déménagement dans une zone non gérée par SEF** : Envoyer un courrier pour clôture.
   - **Si déménagement dans une zone gérée par SEF** : Proposer une reconduction.

6. **Confirmation écrite**
   - Envoyer un email récapitulatif à Mme YYYY avec :
     - Les étapes restantes (ex. : rendez-vous, envoi de documents).
     - La date prévisionnelle de résiliation.

---

**Source(s):**
- Procédure Départ Client | **TRAITER LES DEMANDES DE RÉSILIATION D’ABONNEMENT** (P_NAT_CLI_13, Indice 8)
- Procédure Départ Client | **Traitement du départ immédiat du client** (accès au compteur)

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 5

**Query:** M XXXXXX envoie le message suivant : « Bonsoir, Vous trouverez ci-joint le document notarié demandé. Je vous informe également que mon adresse principale a changé. Je reste dans l’attente de votre confirmation concernant la mise à jour de mon dossier et la  résiliation du contrat concerné. Cordialement, M XXXXXX Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni, M. XXXXXX a transmis :**
- Un **document notarié** (preuve du changement de situation)
- Une **demande de mise à jour d’adresse principale**
- Une **demande de résiliation de contrat**

Voici la **procédure à suivre** pour traiter sa demande :

---

### **1. Mise à jour des données client**
**Actions immédiates :**
- **Vérifier et mettre à jour** les informations suivantes dans le système :
  - Nouvelle adresse principale (obligatoire pour la facturation et les relances).
  - **Email** (MAIL123@MAIL) et **téléphone portable** (si fourni ultérieurement).
- **Confirmer par écrit** (email ou courrier) la prise en compte de la mise à jour.

**Source(s):** [Procédure Départ Client | Traitement du départ immédiat du client – Section 6.2]

---

### **2. Traitement de la résiliation**
**Étapes clés :**
1. **Rappeler au client** (par téléphone ou email) :
   - Il reste **redevable** des abonnements et consommations **jusqu’à la résiliation effective** (facture d’arrêt de compte).
   - La résiliation ne sera effective qu’après **paiement intégral** des sommes dues.
2. **Vérifier la présence d’un successeur** (propriétaire/syndic) :
   - **Si connu** :
     - Créer un **prospect** pour le successeur.
     - Ouvrir un dossier **"Arrivée Client signalée ASI"** (qualifié "information par le prédécesseur").
     - Envoyer un **courrier (1-70)** à l’adresse du site au nom de *"M. et Mme le Nouvel Occupant"* pour inciter à l’abonnement.
   - **Si inconnu** : Noter l’absence de successeur dans le dossier.

**Source(s):** [Procédure Départ Client | TRAITER LES DEMANDES DE RÉSILIATION D'ABONNEMENT] + [Procédure Départ Client | Traitement du départ immédiat du client – Cas 1]

---

### **3. Confirmation finale**
- **Envoyer un email/courrier** à M. XXXXXX pour :
  - **Accuser réception** du document notarié et de la demande de résiliation.
  - **Préciser les étapes restantes** (paiement des factures, délai de traitement).
  - **Fournir un contact dédié** en cas de question.

**Modèle de phrase clé :**
*"Nous vous confirmons la mise à jour de votre adresse et le lancement de la procédure de résiliation. Votre contrat restera actif jusqu’au règlement intégral de votre facture d’arrêt de compte. Un courrier de clôture vous sera adressé sous 15 jours."*

**Source(s):** [Procédure Départ Client | Traitement de la demande d'information sur le départ client]

---

### **4. Points de vigilance**
- **Document notarié** : Archiver une copie numérique dans le dossier client.
- **Délais** : La résiliation n’est effective qu’après **paiement complet** et **émission de la facture d’arrêt**.
- **Successeur** : Si le client ne fournit pas les coordonnées du successeur, **relancer une fois** avant clôture.

**Besoin d’approfondissement ?** Consulter l’**Annexe 10.1 (Schéma)** pour le processus complet.

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 6

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Je vous transmets ce jour le certificat d’hérédité demandé dans le cadre du dossier en cours. Je reste à votre disposition pour toute information complémentaire. Cordialement, M XXXXXX Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni**, M. XXXXXX vous transmet un **certificat d’hérédité** dans le cadre d’un dossier de succession en cours. Voici la procédure à suivre pour traiter cette demande conformément aux règles métiers en vigueur.

---

### **1. Vérification des éléments reçus**
- **Document transmis** : Certificat d’hérédité (à joindre au dossier client).
- **Contexte** : Succession en cours (Cas 1 du rappel métier).
- **Action immédiate** :
  - **Accuser réception** du mail en remerciant M. XXXXXX pour l’envoi du document.
  - **Vérifier la complétude** du certificat (mention des héritiers, signature, cachet de l’autorité compétente, etc.).
  - **Conserver une copie** dans le dossier client (outil ILIADE ou GED).

---

### **2. Procédure à appliquer (Cas 1 : Succession en cours)**
**Règles clés** :
- **Ne pas modifier** le titulaire du contrat ni le RIB dans l’immédiat.
- **Ne pas résilier** le contrat au nom du prédécesseur (sauf demande explicite des héritiers).
- **Créer un nouvel interlocuteur** dans ILIADE pour le successeur (si ce n’est pas déjà fait).

**Étapes détaillées** :
1. **Dans ILIADE** :
   - Ajouter une ligne **"Titulaire"** dans les caractéristiques de l’acte pour identifier le successeur.
   - **Ne pas supprimer** l’ancien titulaire (historique à conserver).
   - Mettre à jour les coordonnées du successeur (mail, téléphone) si fournies.

2. **Traitement du dossier** :
   - **Classer le certificat d’hérédité** dans la section "Pièces justificatives" du dossier.
   - **Mettre le dossier en attente** de validation finale (succession terminée).
   - **Informer le client** que la mutation définitive interviendra après réception de l’acte de notoriété ou du jugement d’hérédité (selon les cas).

3. **Communication avec M. XXXXXX** :
   - **Exemple de réponse** :
     > *"Bonjour M. XXXXXX,
     > Nous accusons réception de votre certificat d’hérédité, que nous avons bien intégré à votre dossier. Conformément à notre procédure, la mutation du contrat au nom du successeur sera finalisée une fois la succession terminée (acte de notoriété ou jugement d’hérédité requis). Nous restons à votre disposition pour toute question.
     > Cordialement, [Votre nom]."*

---

### **3. Points de vigilance**
- **À ne pas faire** :
  - Modifier le RIB ou le titulaire sans document définitif (risque de fraude ou d’erreur).
  - Résilier le contrat sans accord écrit des héritiers.
- **À faire** :
  - **Relancer** M. XXXXXX si le certificat est incomplet (ex. : absence de signature).
  - **Escalader** au service juridique si le dossier présente des particularités (ex. : héritiers mineurs, désaccord entre héritiers).

---

### **4. Prochaines étapes**
- **Si la succession est terminée** (Cas 2) :
  - Demander un **acte de notoriété** ou un **jugement d’hérédité** pour finaliser la mutation.
  - Mettre à jour le titulaire et le RIB dans ILIADE.
- **Si la succession est toujours en cours** :
  - Maintenir le dossier en attente et suivre les relances trimestrielles.

---

**Source(s)** :
- [20250610- Rappel métier - Succession et modification titulaire v4 | Cas 1 : la succession est en cours]
- [20250610- Rappel métier - Succession et modification titulaire v4 | Mutation]

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20250610- Rappel métier - Succession et modification titulaire v4

---

## Question 7

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Je vous contacte suite à une procédure de recouvrement, alors que je n’occupe plus le  logement depuis décembre 2024. Le contrat n’ayant pas été résilié, je vous transmets une attestation de mon ancien propriétaire  confirmant mon départ. Je vous remercie de bien vouloir procéder à la résiliation du contrat. Cordialement, M XXXXXX Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni**, M. XXXXXX demande la résiliation de son contrat d'abonnement en justifiant son départ du logement depuis décembre 2024 via une attestation de son ancien propriétaire. Voici la procédure à suivre pour traiter sa demande :

---

### **1. Vérification des éléments fournis**
- **Attestation du propriétaire** :
  - Vérifier que le document confirme bien :
    - La date de départ effective (décembre 2024).
    - La remise des clés ou la réalisation d’un état des lieux de sortie.
  - **Si l’attestation est valide** : Passer à l’étape 2.
  - **Si l’attestation est incomplète** : Demander un document complémentaire (ex. : état des lieux signé, preuve de remise des clés).

---

### **2. Résiliation du contrat**
- **Date de résiliation** :
  - **La date de fin de contrat doit correspondre à la date de sortie effective** (décembre 2024, selon l’attestation).
  - Le client reste redevable des consommations et de l’abonnement **jusqu’à cette date**.
- **Actions à réaliser** :
  1. **Clôturer le contrat** dans le système avec la date de décembre 2024.
  2. **Générer une facture d’arrêt de compte** incluant :
     - Les consommations jusqu’à la date de résiliation.
     - Les éventuels frais de régularisation (ex. : solde de tout compte).
  3. **Ouvrir un Dossier ASI** (Abandon de Site par l’Installation) avec la branche **"Logement vacant"** pour signaler la vacance du logement.

---

### **3. Communication avec le client**
- **Accusé de réception** :
  - Envoyer un mail à **MAIL123@MAIL** pour confirmer :
    - La prise en compte de sa demande.
    - La date de résiliation retenue (décembre 2024).
    - Les prochaines étapes (envoi de la facture d’arrêt de compte sous X jours).
- **Exemple de message** :
  > *"Bonjour M. XXXXXX,
  > Nous accusons réception de votre demande de résiliation et de l’attestation de départ fournie. Votre contrat sera clôturé à la date du **décembre 2024**, conformément à votre départ effectif. La facture d’arrêt de compte vous sera adressée sous 10 jours ouvrés.
  > Cordialement, [Votre Centre de Relation Client]."*

---

### **4. Cas particuliers à vérifier**
- **Procédure de recouvrement en cours** :
  - Si des relances ont été envoyées **après décembre 2024**, les annuler ou les ajuster.
  - Vérifier que le client n’est pas facturé pour des périodes postérieures à son départ.
- **Accès au compteur** :
  - Si le logement est vacant, une **intervention pour boucher à clé** peut être programmée (cf. procédure nationale).

---

### **5. Suivi post-résiliation**
- **Facture d’arrêt de compte** :
  - Vérifier que la régularisation (solde créditeur/débiteur) est bien incluse.
  - Envoyer la facture par mail (avec copie à l’ancien propriétaire si nécessaire).
- **Dossier ASI** :
  - S’assurer que le dossier est bien ouvert et transmis à la branche concernée.

---

**Source(s)** :
- Procédure Nationale **"TRAITER LES DEMANDES DE RÉSILIATION D’ABONNEMENT"** (Réf : P_NAT_CLI_13, Section "Date de résiliation du contrat").
- Procédure **"DÉPART CLIENT"** (Page 1/12, Paragraphe "Logement vacant").

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 2
- **Contexte:** ~1018 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 8

**Query:** Mme YYYY envoie le message suivant : « Bonjour, Veuillez trouver en pièce jointe la photo du compteur afin de permettre la mise en service du  contrat au nom du nouvel occupant. Je vous remercie de bien vouloir procéder à l’ouverture du contrat. Cordialement, Mme YYYY Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni, Mme YYYY demande la mise en service d'un contrat pour un nouvel occupant en fournissant une photo du compteur. Voici la procédure à suivre et les éléments à vérifier :**

---

### **1. Vérification des conditions d'enregistrement**
**Contexte :**
- La demande inclut une **photo du compteur** (relevé d'index fourni par le client).
- **Pas de relève physique effectuée** par le client ou le technicien à ce stade.

**Analyse des cas possibles (selon la procédure) :**
- **Si le relevé fourni > 10 m³ par rapport à l'index de l'ancien occupant** :
  - **Un RDV de relève est obligatoire**, **frais de déplacement à la charge du centre** (notre responsabilité).
- **Si le relevé est incohérent avec l'historique de consommation** (ex. : consommation anormalement basse/haute) :
  - **Un RDV de relève est obligatoire**, **frais de déplacement à la charge du client**.
- **Si le relevé est cohérent et < 10 m³ de différence** :
  - **Pas de RDV nécessaire** → **Mise en service possible sur la base de la photo**.

---

### **2. Étapes à suivre pour traiter la demande**
1. **Vérifier la cohérence du relevé fourni** :
   - Comparer l'index de la photo avec :
     - L'**index de départ de l'ancien occupant** (disponible dans TPMC/Odyssee).
     - L'**historique de consommation** du site (moyenne annuelle, saisonnalité).
   - **Outils** :
     - TPMC (Tableau de bord client).
     - Lien *"Base des services"* dans Odyssee (vue 360°).

2. **Décision selon le résultat** :
   - **Relevé cohérent** :
     - **Procéder à l'ouverture du contrat** sans RDV.
     - **Informer Mme YYYY** par mail :
       > *"Votre demande de mise en service a été traitée sur la base du relevé fourni. Le contrat est désormais actif. Vous recevrez votre première facture sous [X] jours. Cordialement, [Votre service]."*
   - **Relevé incohérent ou > 10 m³** :
     - **Planifier un RDV de relève** (préciser les frais selon le cas).
     - **Informer Mme YYYY** :
       > *"Votre relevé présente une incohérence avec nos données. Un technicien interviendra le [date] pour effectuer une relève. [Frais à votre charge/à notre charge selon le cas]. Merci de confirmer votre disponibilité. Cordialement, [Votre service]."*

3. **Cas particulier : Photo illisible ou manquante** :
   - **Demander un nouveau envoi** :
     > *"La photo transmise n'est pas exploitable. Merci de nous renvoyer un cliché net du compteur (numéros visibles, sans reflet). Cordialement, [Votre service]."*

---

### **3. Points de vigilance**
- **Délai de traitement** :
  - **Mise en service sous 24h** si relevé cohérent.
  - **RDV sous 48h** si intervention nécessaire.
- **Communication** :
  - **Toujours confirmer par écrit** (mail) les actions engagées et les éventuels frais.
- **Preuves** :
  - **Archiver la photo du compteur** dans le dossier client (Odyssee/TPMC).

---

**Source(s) :**
- *Procédure nationale Arrivée Client V8* | Sections :
  - *"Conditions d’enregistrement d’un établissement public"* (cas "pas de relève").
  - *"Schéma des aides à la décision arrivée client"* (critères de cohérence).
  - *"Traitement de l’arrivée client le jour de son emménagement"* (étapes opérationnelles).

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procedure nationale Arrivée Client V8

---

## Question 9

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Nous avons vendu le bien immobilier et la résiliation du contrat a été effectuée en ligne. Vous trouverez en pièce jointe la photo du compteur réalisée avant la vente. Nous restons à votre disposition pour toute régularisation éventuelle. Cordialement, M XXXXXX Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni, M. XXXXXX informe de la vente de son bien immobilier et de la résiliation en ligne de son contrat, tout en transmettant une photo du compteur. Voici la marche à suivre pour traiter cette demande :**

---

### **1. Vérification de la résiliation en ligne**
- **Confirmer** que la résiliation a bien été effectuée via le canal en ligne (site web/portail client).
  - Si **oui** : Passer à l’étape 2.
  - Si **non** : Inviter le client à finaliser la résiliation en ligne ou proposer une assistance (ex. : lien vers la procédure).

---

### **2. Traitement de la pièce jointe (photo du compteur)**
- **Vérifier la conformité de la photo** :
  - **Index lisible** (chiffres clairs, sans flou).
  - **Date visible** (doit correspondre à la période de vente).
  - **Compteur identifiable** (numéro de série ou localisation si possible).
- **Si la photo est valide** :
  - **Enregistrer l’index** dans le système (fiche G2 ou outil dédié).
  - **Déclencher la facture d’arrêt de compte** (régularisation des consommations jusqu’à la date de la photo).
- **Si la photo est invalide** :
  - **Demander une nouvelle photo** ou proposer un **rendez-vous d’intervention** (si le compteur est accessible).
  - **Rappeler au client** : *Sans accès au compteur, le contrat reste actif et il reste redevable des factures.*

---

### **3. Clôture du contrat et régularisation**
- **Finaliser la résiliation** :
  - Vérifier que la **fiche G2** (retour d’intervention) est remplie dans le système.
  - **Envoyer la facture d’arrêt de compte** (incluant la régularisation des consommations).
- **Informer le client** :
  - **Délai de traitement** : La facture sera émise sous **X jours** (préciser selon la procédure interne).
  - **Paiement** : Le solde doit être réglé pour clôturer définitivement le contrat.

---

### **4. Gestion de l’assurance (si applicable)**
- **Vérifier** si le client avait un contrat d’assurance avec **Homeserve** ou un partenaire.
  - **Si oui** :
    - **Proposer** la clôture du contrat ou son transfert (si le nouveau logement est dans une commune gérée par SEF).
    - **Envoyer un courrier** à Homeserve pour formaliser la demande (modèle disponible dans la procédure).
  - **Si non** : Ignorer cette étape.

---

### **5. Communication finale avec le client**
**Exemple de réponse structurée** :
---
**Objet** : Confirmation de résiliation – Contrat [N° de contrat]

Bonjour M. XXXXXX,

Nous accusons réception de votre demande de résiliation suite à la vente de votre bien. Voici les prochaines étapes :

1. **Photo du compteur** : Votre index a bien été enregistré. La facture d’arrêt de compte sera émise sous **5 jours ouvrés** et inclura la régularisation des consommations jusqu’au [date de la photo].
2. **Assurance** : Votre contrat avec Homeserve a été signalé pour clôture. Vous recevrez un courrier de confirmation sous 10 jours.
3. **Dernières démarches** :
   - Veuillez régler le solde de la facture d’arrêt de compte pour finaliser la clôture.
   - Conservez une copie de cette facture pour votre dossier de vente.

Pour toute question, notre service reste à votre disposition au [numéro de contact].

Cordialement,
[Votre nom/prénom]
[Poste]
[Coordonnées du centre de relation client]
---

---

### **6. Points de vigilance**
- **Délais** : Respecter les échéances de traitement (ex. : envoi de la facture sous 5 jours).
- **Preuves** : Archiver la photo du compteur et la confirmation de résiliation dans le dossier client.
- **Relances** : Si le client ne répond pas aux demandes de régularisation, **escalader** au service recouvrement.

---

**Source(s)** :
- Procédure Départ Client | **Traitement de la demande de résiliation d’abonnement** (Réf : P_NAT_CLI_13, Indice 8).
- Procédure Départ Client | **Traitement du départ immédiat du client** (section "Accès au compteur").

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procédure Départ Client

---

## Question 10

**Query:** Mme YYYY envoie le message suivant : « Bonjour, Je vous fais parvenir la photo du compteur d’eau dans le cadre de mon nouveau contrat suite à  emménagement. Je vous remercie d’en accuser réception. Cordialement, Mme YYYY Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni**, Mme YYYY vous envoie une photo de son compteur d’eau dans le cadre de son **nouvel abonnement suite à un emménagement**. Voici les étapes à suivre pour traiter sa demande et accuser réception :

---

### **1. Accuser réception du message**
**Réponse type à envoyer à Mme YYYY** :
> *"Bonjour Mme YYYY,
> Nous accusons réception de votre envoi concernant la photo du compteur d’eau pour votre nouveau contrat. Votre dossier est en cours de traitement et nous reviendrons vers vous sous 48h avec les prochaines étapes.
> Cordialement,
> [Votre prénom/nom]
> Service Client [SUEZ]"*

---

### **2. Traitement du dossier (étapes métiers)**
#### **A. Vérifications et saisies obligatoires**
1. **Vérifier la cohérence de l’index** :
   - Extraire l’index visible sur la photo du compteur.
   - **S’assurer qu’il est lisible et plausible** (ex. : pas de chiffres incohérents comme "999999").

2. **Saisir les données du dossier** :
   - **Type de résidence** (maison, appartement, etc.).
   - **Statut de l’occupant** (propriétaire, locataire, etc.).
   - **Index du compteur** (celui transmis par Mme YYYY).
   - **Date de l’abonnement** (date du jour ou date d’emménagement si précisée).

3. **Vérifier le montant facturable** :
   - **Doit correspondre à** : *"Frais d’accès au service avec déplacement"*.

4. **Saisir un commentaire dans le dossier** :
   - Format obligatoire :
     > *"[Votre nom] – [Date du jour] : Abonnement – Index saisi : [valeur]. ID de compte : [copier l’ID]."*

5. **Lancer la campagne "Résiliation compte"** (si applicable) :
   - **Uniquement si** un ancien contrat existe pour le même logement (ex. : ancien occupant).

---

#### **B. Points de vigilance**
- **Photo du compteur** :
  - Vérifier que la photo est **nette et complète** (numéros lisibles, matricule du compteur visible si possible).
  - **Si illisible** : Demander une nouvelle photo à Mme YYYY via un message structuré (voir modèle ci-dessous).
    > *"Bonjour Mme YYYY,
    > La photo transmise ne nous permet pas de lire clairement l’index du compteur. Pourriez-vous nous en envoyer une nouvelle, en veillant à ce que :
    > - Les chiffres soient bien visibles.
    > - Le matricule du compteur (situé en haut) soit inclus.
    > Merci pour votre retour.
    > Cordialement, [Votre nom]."*

- **Cohérence des données** :
  - Croiser l’index avec les **anciens relevés** (si disponibles) pour détecter d’éventuelles anomalies.

---

### **3. Finalisation**
- **Envoyer un email de confirmation** à Mme YYYY une fois le dossier traité (ex. : activation du contrat, montant des frais).
- **Archiver** la photo du compteur dans le dossier client.

---

**Source(s)** :
- [20240711 - Rappel métier - Traitement Arrivée Départ avec AT | Saisir un commentaire dans le dossier]
- [20240711 - Rappel métier - Traitement Arrivée Départ avec AT | Vérifier le montant facturable]
- [20240711 - Rappel métier - Traitement Arrivée Départ avec AT | Vérifier la cohérence de l’index]

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- 20240711 - Rappel métier - Traitement Arrivée Départ avec AT

---

## Question 11

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Sauf erreur, je n’ai pas encore eu de retour concernant ma demande d’ouverture de contrat. Je vous remercie de bien vouloir vérifier l’état d’avancement de mon dossier. Cordialement, M XXXXXX Mail : MAIL123@MAIL Tél : 00123456789 »

### Réponse

**D'après le contexte fourni, M. XXXXXX sollicite un suivi sur sa demande d’ouverture de contrat d’eau/assainissement. Voici la démarche à suivre pour traiter sa requête :**

---

### **1. Vérification préalable du statut du client**
- **Confirmer si M. XXXXXX est un professionnel ou un particulier** :
  - *Demander* : *"Pourriez-vous me préciser si cette demande concerne un logement personnel ou un local professionnel ?"*
  - **Si professionnel** : Récupérer les informations obligatoires (SIREN, code NAF) pour compléter le dossier (cf. [Procédure nationale Arrivée Client V8 | Traitement d’une information par anticipation]).
  - **Si particulier** : Passer directement à l’étape 2.

---

### **2. Identification du dossier**
- **Rechercher le dossier** dans l’outil interne via :
  1. **Coordonnées fournies** : Mail (`MAIL123@MAIL`) ou téléphone (`00123456789`).
  2. **Adresse complète** (à demander si absente) :
     - *"Pour vérifier l’avancement, pourriez-vous me communiquer l’adresse exacte du logement concerné (numéro, rue, code postal, ville) ?"*
     - Préciser : *"S’agit-il d’une maison ou d’un appartement ?"* (cf. [Procédure nationale Arrivée Client V8 | Identification]).

- **Vérifier l’état du branchement** (cf. [Traitement de l’arrivée client le jour de l’emménagement]) :
  - **Si compteur en télérelève** : Demander la **date d’entrée dans les lieux** ou d’achat du bien.
  - **Si dossier DSI/ASI** : Créer un prospect si le dossier est ouvert.
  - **Si dossier DEP** : Annuler l’AT (Accusé de Transmission) si nécessaire.

---

### **3. Informations obligatoires à communiquer**
Conformément à la **loi Hamon** (cf. [Procédure nationale Arrivée Client V8 | PRINCIPES]), informer M. XXXXXX sur :
- **Composition de la facture** (part eau et/ou assainissement).
- **Prix de l’eau** : Frais d’accès au service, abonnement, consommation.
- **Modalités de relève du compteur** (télérelève ou auto-relève).

---

### **4. Actions selon l’état du dossier**
| **Situation**               | **Action**                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| **Dossier en cours**        | - Donner un **délai estimé** de traitement.                                |
|                             | - Proposer un **suivi par email** (ex: "Nous vous enverrons un mail sous 48h"). |
| **Dossier finalisé**        | - Confirmer l’ouverture du contrat.                                        |
|                             | - Envoyer un **récapitulatif** (numéro de contrat, date de première facture). |
| **Dossier non trouvé**      | - **Créer un nouveau dossier** avec les informations recueillies.          |
|                             | - Vérifier si un **email automatique** (TSME) a été envoyé le jour de l’emménagement. |

---

### **5. Clôture de l’échange**
- **Proposer une assistance complémentaire** :
  *"Souhaitez-vous que nous vous envoyions un lien pour suivre votre consommation en ligne ?"*
- **Remercier et confirmer** :
  *"Nous avons bien pris en compte votre demande et vous tiendrons informé(e) sous [délai]. N’hésitez pas à nous recontacter pour toute question."*

---

**Source(s)** :
- [Procédure nationale Arrivée Client V8 | Aide au discours arrivée client]
- [Procédure nationale Arrivée Client V8 | Traitement d’une information par anticipation]
- [Procédure nationale Arrivée Client V8 | Traitement de l’arrivée client le jour de l’emménagement]
- Loi Hamon (2014) – [Section PRINCIPES]

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procedure nationale Arrivée Client V8

---

## Question 12

**Query:** M XXXXXX envoie le message suivant : « Bonjour, Comme convenu par téléphone, je vous transmets l’attestation notariale d’acquisition ainsi  que la photo du compteur. Ces éléments sont transmis afin de permettre l’ouverture du contrat. Cordialement, M XXXXXX Mail : MAIL123@MAIL »

### Réponse

**D'après le contexte fourni**, M. XXXXXX a transmis une **attestation notariale d’acquisition** et une **photo du compteur** pour finaliser l’ouverture de son contrat. Voici la procédure à suivre pour traiter sa demande, en conformité avec la **Procédure nationale Arrivée Client V8** :

---

### ### 1. **Vérification des éléments transmis**
- **Attestation notariale** :
  - Confirmer qu’elle prouve bien l’acquisition du logement (propriétaire/locataire).
  - Vérifier la cohérence des informations (adresse, identité du client).
- **Photo du compteur** :
  - **Index de relève** : Extraire le chiffre indiqué sur la photo.
  - **Qualité de la photo** : Doit être lisible (pas de flou, lumière suffisante).
  - **État du compteur** : Vérifier l’absence de signes de fraude ou de détérioration.

---

### ### 2. **Analyse de la relève transmise**
**Règles applicables** (extrait de la procédure) :
1. **Si l’index > 10 m³ par rapport au relevé de l’ancien occupant** :
   - **Action** : Planifier un **RDV de contrôle** (frais de déplacement **à la charge du centre**).
   - **Justification** : Risque de fuite ou erreur de transmission.
2. **Si l’index est incohérent avec l’historique de consommation** (ex. : consommation anormalement basse/haute) :
   - **Action** : Planifier un **RDV de contrôle** (frais de déplacement **à la charge du client**).
   - **Exemple** : Un logement vacant depuis 6 mois avec une relève à 0 m³ est cohérent ; une relève à 50 m³ pour un studio occupé 1 mois ne l’est pas.
3. **Si la relève est absente ou illisible** :
   - **Action** : Demander une **nouvelle photo** au client (modèle de message ci-dessous).

---

### ### 3. **Étapes de traitement**
1. **Enregistrer les documents** dans le système (Odyssee, TPMC, ou base des services) :
   - Lier l’attestation et la photo au dossier client.
   - Noter l’index de relève dans le champ dédié.
2. **Évaluer la cohérence de la relève** :
   - Comparer avec le **relevé de l’ancien occupant** (disponible via la vue 360° d’Odyssee).
   - Croiser avec l’**historique de consommation** du logement (si disponible).
3. **Décision** :
   - **Si OK** : Valider l’ouverture du contrat et envoyer un accusé de réception (modèle ci-dessous).
   - **Si RDV nécessaire** : Contacter le client pour planifier le contrôle (préciser les frais le cas échéant).
   - **Si photo illisible** : Relancer le client pour une nouvelle transmission.

---

### ### 4. **Modèles de communication**
#### **A. Accusé de réception (si documents valides)**
> *Objet : Confirmation de réception de vos documents – Ouverture de contrat*
> *Bonjour M. XXXXXX,*
> *Nous accusons réception de votre attestation notariale et de la photo de votre compteur. Votre demande d’ouverture de contrat est en cours de traitement. Vous recevrez sous 48h un email de confirmation avec les détails de votre abonnement.*
> *Cordialement,*
> *[Votre nom] / Service Relation Client*

#### **B. Demande de nouvelle photo (si illisible)**
> *Objet : Complément nécessaire pour votre contrat*
> *Bonjour M. XXXXXX,*
> *Nous avons bien reçu votre photo de compteur, mais celle-ci n’est pas suffisamment lisible pour enregistrer votre relève. Pourriez-vous nous transmettre une nouvelle photo en veillant à ce que :*
> *- Le cadran soit bien éclairé et net.*
> *- Les chiffres de l’index soient clairement visibles.*
> *Merci de nous renvoyer ce document par retour de mail pour finaliser votre dossier.*
> *Cordialement,*
> *[Votre nom]*

#### **C. Information RDV (si frais à la charge du client)**
> *Objet : RDV de contrôle pour votre compteur – Frais applicables*
> *Bonjour M. XXXXXX,*
> *Suite à l’analyse de votre relève, un RDV de contrôle est nécessaire pour valider votre index. Ce RDV engendrera des **frais de déplacement de [montant] €**, qui seront facturés sur votre première facture.*
> *Voici les créneaux disponibles : [proposer 2-3 dates/horaires]. Merci de nous confirmer votre choix par retour de mail.*
> *Cordialement,*
> *[Votre nom]*

---

### ### 5. **Points de vigilance**
- **Délai** : Traiter la demande sous **48h** pour respecter les engagements clients.
- **Preuves** : Conserver les documents dans le dossier client (traçabilité en cas de litige).
- **Frais** : **Ne jamais facturer de frais sans en informer le client au préalable** (conformément à la procédure).

---

**Source(s)** :
- *Procédure nationale Arrivée Client V8* | Sections :
  - "Conditions d’enregistrement d’un établissement public" (p. [non précisée]).
  - "Schémas des aides à la décision arrivée client" (critères de relève).
- *Base des services Odyssee* (vue 360° pour historique de consommation).

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 3
- **Contexte:** ~1527 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procedure nationale Arrivée Client V8

---

## Question 13

**Query:** Mme YYYY envoie le message suivant : « Bonjour, Je vous transmets les informations nécessaires à l’ouverture de mon contrat d’eau suite à mon  emménagement intervenu le 06/10/2025. Vous trouverez ci-dessous les informations du compteur ainsi que l’index relevé à la date  d’entrée. Je vous remercie de bien vouloir procéder à la mise en service du contrat. Cordialement, Mme YYYY Mail : MAIL123@MAIL Tél : 00123456789 »

### Réponse

**D'après le contexte fourni, Mme YYYY souhaite ouvrir un contrat d'eau suite à son emménagement le 06/10/2025 et transmet les informations nécessaires (index du compteur). Voici la procédure à suivre pour traiter sa demande :**

---

### **1. Vérification préliminaire**
- **S’assurer que SEF (Service des Eaux de France) est bien en charge du contrat** pour l’adresse concernée :
  - Consulter **TPMC** (outil de gestion des contrats) et/ou la **"base des services"** via le compte client **Odyssee** (écran vue 360°).
  - **Rechercher l’adresse** via :
    - Le **site** (adresse complète).
    - Le **matricule du compteur** (si fourni par Mme YYYY).

---

### **2. Analyse de l’index transmis**
- **Comparer l’index fourni par Mme YYYY** avec :
  - L’**index de relève du prédécesseur** (ancien occupant).
  - L’**historique de consommation** du logement (si disponible).

#### **Cas possibles et actions associées** :
1. **Index cohérent** (pas d’écart significatif) :
   - **Enregistrer l’index** et **valider l’ouverture du contrat** sans intervention supplémentaire.

2. **Index incohérent** (écart > 10 m³ ou non conforme à l’historique) :
   - **Déclencher une intervention de relève** avec :
     - **Frais à notre charge** si :
       - L’index transmis est **supérieur de +10 m³** à celui de l’ancien occupant.
     - **Frais à la charge du client** si :
       - L’index est **incohérent avec l’historique de consommation** (ex. : consommation anormalement basse).

3. **Pas de relève transmise** :
   - **Proposer un RDV de relève** (frais selon les mêmes règles ci-dessus).

---

### **3. Ouverture du contrat**
- **Créer le contrat** dans TPMC avec :
  - Les **coordonnées de Mme YYYY** (mail : MAIL123@MAIL, tél : 00123456789).
  - La **date d’emménagement** (06/10/2025).
  - L’**index validé** (ou en attente de relève).
- **Envoyer un accusé de réception** à Mme YYYY (par mail ou SMS) avec :
  - La **confirmation de prise en compte** de sa demande.
  - Les **étapes suivantes** (ex. : date de mise en service, RDV si applicable).
  - Les **coordonnées du service client** pour suivi.

---

### **4. Suivi post-ouverture**
- **Vérifier la cohérence des données** sous 48h (index, adresse, coordonnées).
- **Informer Mme YYYY** en cas de :
  - **Retard** dans la mise en service.
  - **Frais supplémentaires** (si applicable, avec justification).

---

### **Points d’attention**
- **Urgence** : Si Mme YYYY signale un besoin immédiat en eau, **prioriser le traitement** et proposer une solution temporaire (ex. : mise en service sous 24h avec relève ultérieure).
- **Documentation** : Joindre le **livret d’accueil** (mentionné dans la procédure) au premier courrier/facture.

---

**Source(s)** :
- *Procédure nationale Arrivée Client V8* | Sections :
  - **6.2 Traitement de l’arrivée du client le jour de son emménagement** (Page 6/18).
  - **Schémas des aides à la décision arrivée client** (Conditions de relève incohérente).

### Statistiques

- **Retriever:** VectorRetriever
- **Chunks:** 4
- **Contexte:** ~2036 chars
- **Variantes:** 3

### Documents Utilisés

**PRIMARY:**

- Procedure nationale Arrivée Client V8

---

