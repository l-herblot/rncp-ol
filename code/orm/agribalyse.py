from sqlalchemy import Column, Integer, String, Float, Numeric

from models.database import Base


class AgribalyseDataSynthese(Base):
    __tablename__ = "agribalyse_data_synthese"
    code_agb = Column(Integer, primary_key=True)
    code_ciqal = Column(Integer)
    groupe_aliment = Column(String(255))
    sous_groupe_aliment = Column(String(255))
    nom_produit_fr = Column(String(255))
    lci_name = Column(String(255))
    code_saison = Column(Integer)
    code_avion = Column(Integer)
    livraison = Column(String(255))
    materiau_emballage = Column(String(255))
    preparation = Column(String(255))
    dqr = Column(Float)
    score_ef_3_1_mpt_kg = Column(Float)
    changement_climatique_kg_co2_eq = Column(Float)
    appauvrissement_couche_ozone_kg_cvc11_eq = Column(Float)
    rayonnements_ionisants_kbq_u235_eq = Column(Float)
    formation_photochimique_ozone_kg_nmvoc_eq = Column(Float)
    particules_fines_disease_inc_kg = Column(Float)
    effets_toxicologiques_non_cancerogenes_ctuh_kg = Column(Float)
    effets_toxicologiques_cancerogenes_ctuh_kg = Column(Float)
    acidification_terrestre_eaux_douces_mol_h_eq_kg = Column(Float)
    eutrophisation_eaux_douces_kg_p_eq_kg = Column(Float)
    eutrophisation_marine_kg_n_eq_kg = Column(Float)
    eutrophisation_terrestre_mol_n_eq_kg = Column(Float)
    ecotoxicite_aquatiques_eau_douce_ctue_kg = Column(Float)
    utilisation_sol_pt_kg = Column(Float)
    epuisement_ressources_eau_m3_depriv_kg = Column(Float)
    epuisement_ressources_energetiques_mj_kg = Column(Float)
    epuisement_ressources_mineraux_kg_sb_eq_kg = Column(Float)

    def __init__(
        self,
        code_agb,
        code_ciqal,
        groupe_aliment,
        sous_groupe_aliment,
        nom_produit_fr,
        lci_name,
        code_saison,
        code_avion,
        livraison,
        materiau_emballage,
        preparation,
        dqr,
        score_ef_3_1_mpt_kg,
        changement_climatique_kg_co2_eq,
        appauvrissement_couche_ozone_kg_cvc11_eq,
        rayonnements_ionisants_kbq_u235_eq,
        formation_photochimique_ozone_kg_nmvoc_eq,
        particules_fines_disease_inc_kg,
        effets_toxicologiques_non_cancerogenes_ctuh_kg,
        effets_toxicologiques_cancerogenes_ctuh_kg,
        acidification_terrestre_eaux_douces_mol_h_eq_kg,
        eutrophisation_eaux_douces_kg_p_eq_kg,
        eutrophisation_marine_kg_n_eq_kg,
        eutrophisation_terrestre_mol_n_eq_kg,
        ecotoxicite_aquatiques_eau_douce_ctue_kg,
        utilisation_sol_pt_kg,
        epuisement_ressources_eau_m3_depriv_kg,
        epuisement_ressources_energetiques_mj_kg,
        epuisement_ressources_mineraux_kg_sb_eq_kg,
    ):
        self.code_agb = code_agb
        self.code_ciqal = code_ciqal
        self.groupe_aliment = groupe_aliment
        self.sous_groupe_aliment = sous_groupe_aliment
        self.nom_produit_fr = nom_produit_fr
        self.lci_name = lci_name
        self.code_saison = code_saison
        self.code_avion = code_avion
        self.livraison = livraison
        self.materiau_emballage = materiau_emballage
        self.preparation = preparation
        self.dqr = dqr
        self.score_ef_3_1_mpt_kg = score_ef_3_1_mpt_kg
        self.changement_climatique_kg_co2_eq = changement_climatique_kg_co2_eq
        self.appauvrissement_couche_ozone_kg_cvc11_eq = (
            appauvrissement_couche_ozone_kg_cvc11_eq
        )
        self.rayonnements_ionisants_kbq_u235_eq = (
            rayonnements_ionisants_kbq_u235_eq
        )
        self.formation_photochimique_ozone_kg_nmvoc_eq = (
            formation_photochimique_ozone_kg_nmvoc_eq
        )
        self.particules_fines_disease_inc_kg = particules_fines_disease_inc_kg
        self.effets_toxicologiques_non_cancerogenes_ctuh_kg = (
            effets_toxicologiques_non_cancerogenes_ctuh_kg
        )
        self.effets_toxicologiques_cancerogenes_ctuh_kg = (
            effets_toxicologiques_cancerogenes_ctuh_kg
        )
        self.acidification_terrestre_eaux_douces_mol_h_eq_kg = (
            acidification_terrestre_eaux_douces_mol_h_eq_kg
        )
        self.eutrophisation_marine_kg_n_eq_kg = (
            eutrophisation_marine_kg_n_eq_kg
        )
        self.eutrophisation_eaux_douces_kg_p_eq_kg = (
            eutrophisation_eaux_douces_kg_p_eq_kg
        )
        self.eutrophisation_terrestre_mol_n_eq_kg = (
            eutrophisation_terrestre_mol_n_eq_kg
        )
        self.ecotoxicite_aquatiques_eau_douce_ctue_kg = (
            ecotoxicite_aquatiques_eau_douce_ctue_kg
        )
        self.utilisation_sol_pt_kg = utilisation_sol_pt_kg
        self.epuisement_ressources_eau_m3_depriv_kg = (
            epuisement_ressources_eau_m3_depriv_kg
        )
        self.epuisement_ressources_energetiques_mj_kg = (
            epuisement_ressources_energetiques_mj_kg
        )
        self.epuisement_ressources_mineraux_kg_sb_eq_kg = (
            epuisement_ressources_mineraux_kg_sb_eq_kg
        )


class AgribalyseDataByIngredient(Base):
    __tablename__ = "agribalyse_data_by_ingredient"
    ciqual_agb = Column(String(10), primary_key=True)
    ciqual_code = Column(String(10))
    groupe_aliment = Column(String(255))
    sous_groupe_aliment = Column(String(255))
    nom_francais = Column(String(255))
    lci_name = Column(String(255))
    ingredients = Column(String(255))
    ef_mpt = Column(Numeric(10, 2))
    changement_climatique = Column(Numeric(10, 2))
    appauvrissement_couche_ozone = Column(Numeric(10, 2))
    rayonnements_ionisants = Column(Numeric(10, 2))
    formation_photochimique_ozone = Column(Numeric(10, 2))
    particules_fines_disease = Column(Numeric(10, 2))
    effets_toxicologiques_non_cancerogenes = Column(Numeric(10, 2))
    effets_toxicologiques_cancerogenes = Column(Numeric(10, 2))
    acidification_terrestre_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_marine = Column(Numeric(10, 2))
    eutrophisation_terrestre = Column(Numeric(10, 2))
    ecotoxicite_ecosystemes_aquatiques = Column(Numeric(10, 2))
    utilisation_du_sol = Column(Numeric(10, 2))
    epuisement_ressources_eau = Column(Numeric(10, 2))
    epuisement_ressources_energetiques = Column(Numeric(10, 2))
    epuisement_ressources_mineraux = Column(Numeric(10, 2))

    def __init__(
        self,
        ciqual_agb,
        ciqual_code,
        groupe_aliment,
        sous_groupe_aliment,
        nom_francais,
        lci_name,
        ingredients,
        ef_mpt,
        changement_climatique,
        appauvrissement_couche_ozone,
        rayonnements_ionisants,
        formation_photochimique_ozone,
        particules_fines_disease,
        effets_toxicologiques_non_cancerogenes,
        effets_toxicologiques_cancerogenes,
        acidification_terrestre_eaux_douces,
        eutrophisation_eaux_douces,
        eutrophisation_marine,
        eutrophisation_terrestre,
        ecotoxicite_ecosystemes_aquatiques,
        utilisation_du_sol,
        epuisement_ressources_eau,
        epuisement_ressources_energetiques,
        epuisement_ressources_mineraux,
    ):
        self.ciqual_agb = ciqual_agb
        self.ciqual_code = ciqual_code
        self.groupe_aliment = groupe_aliment
        self.sous_groupe_aliment = sous_groupe_aliment
        self.nom_francais = nom_francais
        self.lci_name = lci_name
        self.ingredients = ingredients
        self.ef_mpt = ef_mpt
        self.changement_climatique = changement_climatique
        self.appauvrissement_couche_ozone = appauvrissement_couche_ozone
        self.rayonnements_ionisants = rayonnements_ionisants
        self.formation_photochimique_ozone = formation_photochimique_ozone
        self.particules_fines_disease = particules_fines_disease
        self.effets_toxicologiques_non_cancerogenes = (
            effets_toxicologiques_non_cancerogenes
        )
        self.effets_toxicologiques_cancerogenes = (
            effets_toxicologiques_cancerogenes
        )
        self.acidification_terrestre_eaux_douces = (
            acidification_terrestre_eaux_douces
        )
        self.eutrophisation_eaux_douces = eutrophisation_eaux_douces
        self.eutrophisation_marine = eutrophisation_marine
        self.eutrophisation_terrestre = eutrophisation_terrestre
        self.ecotoxicite_ecosystemes_aquatiques = (
            ecotoxicite_ecosystemes_aquatiques
        )
        self.utilisation_du_sol = utilisation_du_sol
        self.epuisement_ressources_eau = epuisement_ressources_eau
        self.epuisement_ressources_energetiques = (
            epuisement_ressources_energetiques
        )
        self.epuisement_ressources_mineraux = epuisement_ressources_mineraux


class AgribalyseDataFrBio(Base):
    __tablename__ = "agribalyse_data_fr_bio"
    nom_produit_francais = Column(String(255), primary_key=True)
    lci_name = Column(String(255))
    categorie = Column(String(255))
    ef_mpt = Column(Numeric(10, 2))
    changement_climatique = Column(Numeric(10, 2))
    appauvrissement_couche_ozone = Column(Numeric(10, 2))
    rayonnements_ionisants = Column(Numeric(10, 2))
    formation_photochimique_ozone = Column(Numeric(10, 2))
    particules_fines_disease = Column(Numeric(10, 2))
    effets_toxicologiques_non_cancerogenes = Column(Numeric(10, 2))
    effets_toxicologiques_cancerogenes = Column(Numeric(10, 2))
    acidification_terrestre_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_marine = Column(Numeric(10, 2))
    eutrophisation_terrestre = Column(Numeric(10, 2))
    ecotoxicite_ecosystemes_aquatiques = Column(Numeric(10, 2))
    utilisation_du_sol = Column(Numeric(10, 2))
    epuisement_ressources_eau = Column(Numeric(10, 2))
    epuisement_ressources_energetiques = Column(Numeric(10, 2))
    epuisement_ressources_mineraux = Column(Numeric(10, 2))

    def __init__(
        self,
        nom_produit_francais,
        lci_name,
        categorie,
        ef_mpt,
        changement_climatique,
        appauvrissement_couche_ozone,
        rayonnements_ionisants,
        formation_photochimique_ozone,
        particules_fines_disease,
        effets_toxicologiques_non_cancerogenes,
        effets_toxicologiques_cancerogenes,
        acidification_terrestre_eaux_douces,
        eutrophisation_eaux_douces,
        eutrophisation_marine,
        eutrophisation_terrestre,
        ecotoxicite_ecosystemes_aquatiques,
        utilisation_du_sol,
        epuisement_ressources_eau,
        epuisement_ressources_energetiques,
        epuisement_ressources_mineraux,
    ):
        self.nom_produit_francais = nom_produit_francais
        self.lci_name = lci_name
        self.categorie = categorie
        self.ef_mpt = ef_mpt
        self.changement_climatique = changement_climatique
        self.appauvrissement_couche_ozone = appauvrissement_couche_ozone
        self.rayonnements_ionisants = rayonnements_ionisants
        self.formation_photochimique_ozone = formation_photochimique_ozone
        self.particules_fines_disease = particules_fines_disease
        self.effets_toxicologiques_non_cancerogenes = (
            effets_toxicologiques_non_cancerogenes
        )
        self.effets_toxicologiques_cancerogenes = (
            effets_toxicologiques_cancerogenes
        )
        self.acidification_terrestre_eaux_douces = (
            acidification_terrestre_eaux_douces
        )
        self.eutrophisation_eaux_douces = eutrophisation_eaux_douces
        self.eutrophisation_marine = eutrophisation_marine
        self.eutrophisation_terrestre = eutrophisation_terrestre
        self.ecotoxicite_ecosystemes_aquatiques = (
            ecotoxicite_ecosystemes_aquatiques
        )
        self.utilisation_du_sol = utilisation_du_sol
        self.epuisement_ressources_eau = epuisement_ressources_eau
        self.epuisement_ressources_energetiques = (
            epuisement_ressources_energetiques
        )
        self.epuisement_ressources_mineraux = epuisement_ressources_mineraux


class AgribalyseDataFrConv(Base):
    __tablename__ = "agribalyse_data_fr_conv"
    nom_produit_francais = Column(String(255), primary_key=True)
    lci_name = Column(String(255))
    categorie = Column(String(255))
    ef_mpt = Column(Numeric(10, 2))
    changement_climatique = Column(Numeric(10, 2))
    appauvrissement_couche_ozone = Column(Numeric(10, 2))
    rayonnements_ionisants = Column(Numeric(10, 2))
    formation_photochimique_ozone = Column(Numeric(10, 2))
    particules_fines_disease = Column(Numeric(10, 2))
    effets_toxicologiques_non_cancerogenes = Column(Numeric(10, 2))
    effets_toxicologiques_cancerogenes = Column(Numeric(10, 2))
    acidification_terrestre_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_eaux_douces = Column(Numeric(10, 2))
    eutrophisation_marine = Column(Numeric(10, 2))
    eutrophisation_terrestre = Column(Numeric(10, 2))
    ecotoxicite_ecosystemes_aquatiques = Column(Numeric(10, 2))
    utilisation_du_sol = Column(Numeric(10, 2))
    epuisement_ressources_eau = Column(Numeric(10, 2))
    epuisement_ressources_energetiques = Column(Numeric(10, 2))
    epuisement_ressources_mineraux = Column(Numeric(10, 2))

    def __init__(
        self,
        nom_produit_francais,
        lci_name,
        categorie,
        ef_mpt,
        changement_climatique,
        appauvrissement_couche_ozone,
        rayonnements_ionisants,
        formation_photochimique_ozone,
        particules_fines_disease,
        effets_toxicologiques_non_cancerogenes,
        effets_toxicologiques_cancerogenes,
        acidification_terrestre_eaux_douces,
        eutrophisation_eaux_douces,
        eutrophisation_marine,
        eutrophisation_terrestre,
        ecotoxicite_ecosystemes_aquatiques,
        utilisation_du_sol,
        epuisement_ressources_eau,
        epuisement_ressources_energetiques,
        epuisement_ressources_mineraux,
    ):
        self.nom_produit_francais = nom_produit_francais
        self.lci_name = lci_name
        self.categorie = categorie
        self.ef_mpt = ef_mpt
        self.changement_climatique = changement_climatique
        self.appauvrissement_couche_ozone = appauvrissement_couche_ozone
        self.rayonnements_ionisants = rayonnements_ionisants
        self.formation_photochimique_ozone = formation_photochimique_ozone
        self.particules_fines_disease = particules_fines_disease
        self.effets_toxicologiques_non_cancerogenes = (
            effets_toxicologiques_non_cancerogenes
        )
        self.effets_toxicologiques_cancerogenes = (
            effets_toxicologiques_cancerogenes
        )
        self.acidification_terrestre_eaux_douces = (
            acidification_terrestre_eaux_douces
        )
        self.eutrophisation_eaux_douces = eutrophisation_eaux_douces
        self.eutrophisation_marine = eutrophisation_marine
        self.eutrophisation_terrestre = eutrophisation_terrestre
        self.ecotoxicite_ecosystemes_aquatiques = (
            ecotoxicite_ecosystemes_aquatiques
        )
        self.utilisation_du_sol = utilisation_du_sol
        self.epuisement_ressources_eau = epuisement_ressources_eau
        self.epuisement_ressources_energetiques = (
            epuisement_ressources_energetiques
        )
        self.epuisement_ressources_mineraux = epuisement_ressources_mineraux
