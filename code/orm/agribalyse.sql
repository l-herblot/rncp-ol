CREATE TABLE IF NOT EXISTS public.agribalyse_data_synthese
(
    code_agb integer,
    code_ciqal integer,
    groupe_aliment character varying(255) COLLATE pg_catalog."default",
    sous_groupe_aliment character varying(255) COLLATE pg_catalog."default",
    nom_produit_fr character varying(255) COLLATE pg_catalog."default",
    lci_name character varying(255) COLLATE pg_catalog."default",
    code_saison integer,
    code_avion integer,
    livraison character varying(255) COLLATE pg_catalog."default",
    materiau_emballage character varying(255) COLLATE pg_catalog."default",
    preparation character varying(255) COLLATE pg_catalog."default",
    dqr double precision,
    score_ef_3_1_mpt_kg double precision,
    changement_climatique_kg_co2_eq double precision,
    appauvrissement_couche_ozone_kg_cvc11_eq double precision,
    rayonnements_ionisants_kbq_u235_eq double precision,
    formation_photochimique_ozone_kg_nmvoc_eq double precision,
    particules_fines_disease_inc_kg double precision,
    effets_toxicologiques_non_cancerogenes_ctuh_kg double precision,
    effets_toxicologiques_cancerogenes_ctuh_kg double precision,
    acidification_terrestre_eaux_douces_mol_h_eq_kg double precision,
    eutrophisation_eaux_douces_kg_p_eq_kg double precision,
    eutrophisation_marine_kg_n_eq_kg double precision,
    eutrophisation_terrestre_mol_n_eq_kg double precision,
    ecotoxicite_aquatiques_eau_douce_ctue_kg double precision,
    utilisation_sol_pt_kg double precision,
    epuisement_ressources_eau_m3_depriv_kg double precision,
    epuisement_ressources_energetiques_mj_kg double precision,
    epuisement_ressources_mineraux_kg_sb_eq_kg double precision
);

COMMENT ON TABLE public.agribalyse_data_synthese
    IS 'code_agb: Code AGB
code_ciqal: Code CIQUAL
groupe_aliment: Groupe d''aliment
sous_groupe_aliment: Sous-groupe d''aliment
nom_produit_fr: Nom du Produit en Français
lci_name: LCI Name
code_saison: Code saison
code_avion: Code avion
livraison: Livraison
materiau_emballage: Matériau d''emballage
preparation: Préparation
dqr: DQR - Note de qualité de la donnéescore_ef_3_1_mpt_kg: Score unique EF 3.1 mPt/kg de produit
changement_climatique_kg_co2_eq: Changement climatique kg CO2 eq/kg de produit
appauvrissement_couche_ozone_kg_cvc11_eq: Appauvrissement de la couche d''ozone kg CVC11 eq/kg de produit
rayonnements_ionisants_kBq_u235_eq: Rayonnements ionisants kBq U-235 eq/kg de produit
formation_photochimique_ozone_kg_nmvoc_eq: Formation photochimique d''ozone kg NMVOC eq/kg de produit
particules_fines_disease_inc_kg: Particules fines disease inc./kg de produit
effets_toxicologiques_non_cancerogenes_ctuh_kg: Effets toxicologiques sur la santé humaine : substances non-cancérogènes CTUh/kg de produit
effets_toxicologiques_cancerogenes_ctuh_kg: Effets toxicologiques sur la santé humaine : substances cancérogènes CTUh/kg de produit
acidification_terrestre_eaux_douces_mol_h_eq_kg: Acidification terrestre et eaux douces mol H+ eq/kg de produit
eutrophisation_eaux_douces_kg_p_eq_kg: Eutrophisation eaux douces kg P eq/kg de produit
eutrophisation_marine_kg_n_eq_kg: Eutrophisation marine kg N eq/kg de produit
eutrophisation_terrestre_mol_n_eq_kg: Eutrophisation terrestre mol N eq/kg de produit
ecotoxicite_aquatiques_eau_douce_ctue_kg: Écotoxicité pour écosystèmes aquatiques d''eau douce CTUe/kg de produit
utilisation_sol_pt_kg: Utilisation du sol Pt/kg de produit
epuisement_ressources_eau_m3_depriv_kg: Épuisement des ressources eau m3 depriv./kg de produit
epuisement_ressources_energetiques_mj_kg: Épuisement des ressources énergétiques MJ/kg de produit
epuisement_ressources_mineraux_kg_sb_eq_kg: Épuisement des ressources minéraux kg Sb eq/kg de produit ';

CREATE TABLE IF NOT EXISTS public.agribalyse_data_by_ingredient
(
    ciqual_agb character varying(10) COLLATE pg_catalog."default",
    ciqual_code character varying(10) COLLATE pg_catalog."default",
    groupe_aliment character varying(255) COLLATE pg_catalog."default",
    sous_groupe_aliment character varying(255) COLLATE pg_catalog."default",
    nom_francais character varying(255) COLLATE pg_catalog."default",
    lci_name character varying(255) COLLATE pg_catalog."default",
    ingredients character varying(255) COLLATE pg_catalog."default",
    ef_mpt numeric(10,2),
    changement_climatique numeric(10,2),
    appauvrissement_couche_ozone numeric(10,2),
    rayonnements_ionisants numeric(10,2),
    formation_photochimique_ozone numeric(10,2),
    particules_fines_disease numeric(10,2),
    effets_toxicologiques_non_cancerogenes numeric(10,2),
    effets_toxicologiques_cancerogenes numeric(10,2),
    acidification_terrestre_eaux_douces numeric(10,2),
    eutrophisation_eaux_douces numeric(10,2),
    eutrophisation_marine numeric(10,2),
    eutrophisation_terrestre numeric(10,2),
    ecotoxicite_ecosystemes_aquatiques numeric(10,2),
    utilisation_du_sol numeric(10,2),
    epuisement_ressources_eau numeric(10,2),
    epuisement_ressources_energetiques numeric(10,2),
    epuisement_ressources_mineraux numeric(10,2)
);

CREATE TABLE IF NOT EXISTS public.agribalyse_data_fr_bio
(
    nom_produit_francais character varying(255) COLLATE pg_catalog."default",
    lci_name character varying(255) COLLATE pg_catalog."default",
    categorie character varying(255) COLLATE pg_catalog."default",
    ef_mpt numeric(10,2),
    changement_climatique numeric(10,2),
    appauvrissement_couche_ozone numeric(10,2),
    rayonnements_ionisants numeric(10,2),
    formation_photochimique_ozone numeric(10,2),
    particules_fines_disease numeric(10,2),
    effets_toxicologiques_non_cancerogenes numeric(10,2),
    effets_toxicologiques_cancerogenes numeric(10,2),
    acidification_terrestre_eaux_douces numeric(10,2),
    eutrophisation_eaux_douces numeric(10,2),
    eutrophisation_marine numeric(10,2),
    eutrophisation_terrestre numeric(10,2),
    ecotoxicite_ecosystemes_aquatiques numeric(10,2),
    utilisation_du_sol numeric(10,2),
    epuisement_ressources_eau numeric(10,2),
    epuisement_ressources_energetiques numeric(10,2),
    epuisement_ressources_mineraux numeric(10,2)
);

CREATE TABLE IF NOT EXISTS public.agribalyse_data_fr_conv
(
    nom_produit_francais character varying(255) COLLATE pg_catalog."default",
    lci_name character varying(255) COLLATE pg_catalog."default",
    categorie character varying(255) COLLATE pg_catalog."default",
    ef_mpt numeric(10,2),
    changement_climatique numeric(10,2),
    appauvrissement_couche_ozone numeric(10,2),
    rayonnements_ionisants numeric(10,2),
    formation_photochimique_ozone numeric(10,2),
    particules_fines_disease numeric(10,2),
    effets_toxicologiques_non_cancerogenes numeric(10,2),
    effets_toxicologiques_cancerogenes numeric(10,2),
    acidification_terrestre_eaux_douces numeric(10,2),
    eutrophisation_eaux_douces numeric(10,2),
    eutrophisation_marine numeric(10,2),
    eutrophisation_terrestre numeric(10,2),
    ecotoxicite_ecosystemes_aquatiques numeric(10,2),
    utilisation_du_sol numeric(10,2),
    epuisement_ressources_eau numeric(10,2),
    epuisement_ressources_energetiques numeric(10,2),
    epuisement_ressources_mineraux numeric(10,2)
);

CREATE TABLE IF NOT EXISTS public.agribalyse_data_by_step
(
    code_agb integer,
    code_ciqual integer,
    groupe_aliment character varying(255) COLLATE pg_catalog."default",
    sous_groupe_aliment character varying(255) COLLATE pg_catalog."default",
    nom_produit_francais character varying(255) COLLATE pg_catalog."default",
    lci_name character varying(255) COLLATE pg_catalog."default",
    dqr numeric(10,2),
    ef_agriculture numeric(10,2),
    ef_transformation numeric(10,2),
    ef_emballage numeric(10,2),
    ef_transport numeric(10,2),
    ef_supermarche_distribution numeric(10,2),
    ef_consommation numeric(10,2),
    ef_total numeric(10,2),
    changement_climatique_agriculture numeric(10,2),
    changement_climatique_transformation numeric(10,2),
    changement_climatique_emballage numeric(10,2),
    changement_climatique_transport numeric(10,2),
    changement_climatique_supermarche_distribution numeric(10,2),
    changement_climatique_consommation numeric(10,2),
    changement_climatique_total numeric(10,2),
    appauvrissement_couche_ozone_agriculture numeric(10,2),
    appauvrissement_couche_ozone_transformation numeric(10,2),
    appauvrissement_couche_ozone_emballage numeric(10,2),
    appauvrissement_couche_ozone_transport numeric(10,2),
    appauvrissement_couche_ozone_supermarche_distribution numeric(10,2),
    appauvrissement_couche_ozone_consommation numeric(10,2),
    appauvrissement_couche_ozone_total numeric(10,2),
    rayonnements_ionisants_agriculture numeric(10,2),
    rayonnements_ionisants_transformation numeric(10,2),
    rayonnements_ionisants_emballage numeric(10,2),
    rayonnements_ionisants_transport numeric(10,2),
    rayonnements_ionisants_supermarche_distribution numeric(10,2),
    rayonnements_ionisants_consommation numeric(10,2),
    rayonnements_ionisants_total numeric(10,2),
    formation_photochimique_ozone_agriculture numeric(10,2),
    formation_photochimique_ozone_transformation numeric(10,2),
    formation_photochimique_ozone_emballage numeric(10,2),
    formation_photochimique_ozone_transport numeric(10,2),
    formation_photochimique_ozone_supermarche_distribution numeric(10,2),
    formation_photochimique_ozone_consommation numeric(10,2),
    formation_photochimique_ozone_total numeric(10,2),
    particules_fines_agriculture numeric(10,2),
    particules_fines_transformation numeric(10,2),
    particules_fines_emballage numeric(10,2),
    particules_fines_transport numeric(10,2),
    particules_fines_supermarche_distribution numeric(10,2),
    particules_fines_consommation numeric(10,2),
    particules_fines_total numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_agriculture numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_transformation numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_emballage numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_transport numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_supermarche_distri numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_consommation numeric(10,2),
    effets_toxicologiques_sante_non_cancerogenes_total numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_agriculture numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_transformation numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_emballage numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_transport numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_supermarche_distributi numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_consommation numeric(10,2),
    effets_toxicologiques_sante_cancerogenes_total numeric(10,2),
    acidification_terrestre_eaux_douces_agriculture numeric(10,2),
    acidification_terrestre_eaux_douces_transformation numeric(10,2),
    acidification_terrestre_eaux_douces_emballage numeric(10,2),
    acidification_terrestre_eaux_douces_transport numeric(10,2),
    acidification_terrestre_eaux_douces_supermarche_distribution numeric(10,2),
    acidification_terrestre_eaux_douces_consommation numeric(10,2),
    acidification_terrestre_eaux_douces_total numeric(10,2),
    eutrophisation_eaux_douces_agriculture numeric(10,2),
    eutrophisation_eaux_douces_transformation numeric(10,2),
    eutrophisation_eaux_douces_emballage numeric(10,2),
    eutrophisation_eaux_douces_transport numeric(10,2),
    eutrophisation_eaux_douces_supermarche_distribution numeric(10,2),
    eutrophisation_eaux_douces_consommation numeric(10,2),
    eutrophisation_eaux_douces_total numeric(10,2),
    eutrophisation_marine_agriculture numeric(10,2),
    eutrophisation_marine_transformation numeric(10,2),
    eutrophisation_marine_emballage numeric(10,2),
    eutrophisation_marine_transport numeric(10,2),
    eutrophisation_marine_supermarche_distribution numeric(10,2),
    eutrophisation_marine_consommation numeric(10,2),
    eutrophisation_marine_total numeric(10,2),
    eutrophisation_terrestre_agriculture numeric(10,2),
    eutrophisation_terrestre_transformation numeric(10,2),
    eutrophisation_terrestre_emballage numeric(10,2),
    eutrophisation_terrestre_transport numeric(10,2),
    eutrophisation_terrestre_supermarche_distribution numeric(10,2),
    eutrophisation_terrestre_consommation numeric(10,2),
    eutrophisation_terrestre_total numeric(10,2),
    ecotoxicite_eau_douce_agriculture numeric(10,2),
    ecotoxicite_eau_douce_transformation numeric(10,2),
    ecotoxicite_eau_douce_emballage numeric(10,2),
    ecotoxicite_eau_douce_transport numeric(10,2),
    ecotoxicite_eau_douce_supermarche_distribution numeric(10,2),
    ecotoxicite_eau_douce_consommation numeric(10,2),
    ecotoxicite_eau_douce_total numeric(10,2),
    utilisation_terres_sols_agriculture numeric(10,2),
    utilisation_terres_sols_transformation numeric(10,2),
    utilisation_terres_sols_emballage numeric(10,2),
    utilisation_terres_sols_transport numeric(10,2),
    utilisation_terres_sols_supermarche_distribution numeric(10,2),
    utilisation_terres_sols_consommation numeric(10,2),
    utilisation_terres_sols_total numeric(10,2),
    epuisement_eau_agriculture numeric(10,2),
    epuisement_eau_transformation numeric(10,2),
    epuisement_eau_emballage numeric(10,2),
    epuisement_eau_transport numeric(10,2),
    epuisement_eau_supermarche_distribution numeric(10,2),
    epuisement_eau_consommation numeric(10,2),
    epuisement_eau_total numeric(10,2),
    epuisement_energie_agriculture numeric(10,2),
    epuisement_energie_transformation numeric(10,2),
    epuisement_energie_emballage numeric(10,2),
    epuisement_energie_transport numeric(10,2),
    epuisement_energie_supermarche_distribution numeric(10,2),
    epuisement_energie_consommation numeric(10,2),
    epuisement_energie_total numeric(10,2),
    epuisement_mineraux_agriculture numeric(10,2),
    epuisement_mineraux_transformation numeric(10,2),
    epuisement_mineraux_emballage numeric(10,2),
    epuisement_mineraux_transport numeric(10,2),
    epuisement_mineraux_supermarche_distribution numeric(10,2),
    epuisement_mineraux_consommation numeric(10,2),
    epuisement_mineraux_total numeric(10,2),
    dqr_overall numeric(10,2),
    p numeric(10,2),
    tir numeric(10,2),
    gr numeric(10,2),
    ter numeric(10,2)
);
