# TCR-Epitope-Binding

The recognition of antigens by T-cells and B-cells are a core part of the human immune system's protection against viruses, bateria, and even cancer. This project sets out to predict the binding of T-cells Receptors (TCR) with an Epitope, a part of an antigen that is recognized by the T-cell.

## Data
Three publicly available datasets are used for this project: Adaptive Biotechnologies’ ImmuneCODE database containing putative SARS-CoV-2-specific TCR sequences and their corresponding epitopes [1]; VDJdb: a collection of paired TCR sequences and epitopes from previously published studies, largely in infectious diseases [2]; and McPAS-TCR, a manually curated database of TCR sequences associated with various pathologies and antigens [3]. Here are some sample TCR and Epitope sequence pairs as well as their Antigen type:

![Sample Sequences](https://github.com/el535/TCR-Epitope-Binding/blob/main/Project_Images/Sample_TCR_Epitope_Sequences.JPG)

## Model
Three types of models are created: one baseline Multilayer Perceptron n-gram model, one LSTM model, and two similar Transformer (BERT) models. The LSTM model mimics the model architecture presented in [4], the previous State of the Art model for predicting TCR-Epitope pairs. A table of hyperparameters and model architecture are shown here:

![Table of models](https://github.com/el535/TCR-Epitope-Binding/blob/main/Project_Images/Model_Table.JPG)

Diagrams of the BERT and LSTM models are shown here:
![BERT_LSTM](https://github.com/el535/TCR-Epitope-Binding/blob/main/Project_Images/Model_Diagram.JPG)

## Results
Precision-Recall curves and ROC curves for all models are shown here:
![Curves](https://github.com/el535/TCR-Epitope-Binding/blob/main/Project_Images/Curves.JPG)

The BERT-mini model created performs the best overall, having the best Average Precision and ROC AUC out of all the models. The model also beats the LSTM model that represetnes the previous State of the Art model. Some sample real (left) and BERT-mini model learned (right) epitope sequence motifs are shown here:
![Sequence_Motifs](https://github.com/el535/TCR-Epitope-Binding/blob/main/Project_Images/Sequence_Motifs.JPG)

Two practical applications for this project are identification of molecular (TCR) biomarkers of viral disease or immune response, such as response to vaccination, and identification of tumorspecific TCRs for potential CAR-T cell therapy.

## References
- [1] Nolan S, Vignali M, Klinger M, Dines JN, Kaplan IM, Svejnoha E, Craft T, Boland K, Pesesky M, Gittelman RM, Snyder TM, Gooley CJ, Semprini S, Cerchione C, Mazza M, Delmonte OM, Dobbs K, Carreño-Tarragona G, Barrio S, Sambri V, Martinelli G, Goldman JD, Heath JR, Notarangelo LD, Carlson JM, Martinez-Lopez J, Robins HS. A large-scale database of T-cell receptor beta (TCR) sequences and binding associations from natural and synthetic exposure to SARS-CoV-2. Res Sq [Preprint]. 2020 Aug 4:rs.3.rs-51964. doi: 10.21203/rs.3.rs- 51964/v1. PMID: 32793896; PMCID: PMC7418738
- [2] Shugay M, Bagaev DV, Zvyagin IV, Vroomans RM, Crawford JC, Dolton G, Komech EA, Sycheva AL, Koneva AE, Egorov ES, Eliseev AV, Van Dyk E, Dash P, Attaf M, Rius C, Ladell K, McLaren JE, Matthews KK, Clemens EB, Douek DC, Luciani F, van Baarle D, Kedzierska K, Kesmir C, Thomas PG, Price DA, Sewell AK, Chudakov DM. VDJdb: a curated database of T-cell receptor sequences with known antigen specificity. Nucleic Acids Res. 2018 Jan 4;46(D1):D419-D427. doi: 10.1093/nar/gkx760. PMID: 28977646; PMCID: PMC5753233.
- [3] Tickotsky N, Sagiv T, Prilusky J, Shifrut E, Friedman N. McPAS-TCR: a manually curated catalogue of pathology-associated T cell receptor sequences. Bioinformatics. 2017 Sep 15;33(18):2924-2929. doi:10.1093/bioinformatics/btx286. PMID: 28481982.
- [4] Ido Springer, Hanan Besser, Nili Tickotsky-Moskovitz, Shirit Dvorkin, Yoram Louzoun: Prediction of specific TCR-peptide binding from large dictionaries of TCR-peptide pairs. BioRxiv, 2020.https://doi.org/10.1101/650861
