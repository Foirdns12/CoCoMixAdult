"""Load data for demonstration.

All data used in the demonstration is loaded through this module
to ensure consistency in the features, their type, and their order.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import immo

FEATURES = [
    'obj_yearConstructed',
    'obj_livingSpace',
    'obj_noRooms',
    'obj_numberOfFloors',
    'obj_lotArea',
    'obj_noParkSpaces',
    #'Einwohnerdichte_PLZ',
    'obj_buildingType',
    'obj_cellar',
    'obj_condition',
    'obj_interiorQual',
    'obj_heatingType',
    'obj_regio1',
    'geo_krs',
    #'obj_regio4'
]

VAR_TYPES = [
    'c',
    'c',
    'c',
    'c',
    'c',
    'c',
    #'c',
    'u',
    'u',
    'u',
    'o',
    'u',
    'u',
    'u',
    #'u'
]


TARGET = "obj_purchasePrice"


UNORDERED_CATEGORICAL_VALUES = {
    'obj_buildingType': ['no_information', 'single_family_house', 'multi_family_house', 'semidetached_house',
                         'mid_terrace_house', 'end_terrace_house', 'villa', 'bungalow', 'farmhouse',
                         'castle_manor_house'],
    'obj_cellar': ['y', 'n'],
    'obj_condition': ['no_information', 'mint_condition', 'first_time_use', 'modernized',
                      'fully_renovated', 'first_time_use_after_refurbishment', 'refurbished',
                      'well_kept', 'need_of_renovation', 'negotiable', 'ripe_for_demolition'],
    'obj_heatingType': ['no_information', 'central_heating', 'heat_pump', 'gas_heating', 'floor_heating',
                        'self_contained_central_heating', 'stove_heating', 'oil_heating', 'night_storage_heater',
                        'electric_heating', 'combined_heat_and_power_plant', 'wood_pellet_heating', 'district_heating',
                        'solar_heating'],
    'obj_regio1': ['Saarland', 'Bayern', 'Hessen', 'Nordrhein_Westfalen', 'Baden_Württemberg', 'Hamburg', 'Brandenburg',
                   'Sachsen', 'Schleswig_Holstein', 'Niedersachsen', 'Thüringen', 'Rheinland_Pfalz',
                   'Mecklenburg_Vorpommern', 'Berlin', 'Sachsen_Anhalt', 'Bremen'],
    'geo_krs': ['nordvorpommern_kreis', 'parchim_kreis', 'güstrow_kreis', 'ludwigslust_kreis', 'kempten_allgäu', 'müritz_kreis', 'bad_doberan_kreis', 'rügen_kreis', 'uecker_randow_kreis', 'amberg', 'neustadt_a.d._aisch_bad_windsheim_kreis', 'weiden_in_der_oberpfalz', 'frankfurt_oder', 'stralsund', 'greifswald', 'mecklenburg_strelitz_kreis', 'neustadt_a.d._waldnaab_kreis', 'neubrandenburg', 'wismar', 'neumünster', 'hoyerswerda', 'demmin_kreis','stormarn_kreis', 'kelheim_kreis', 'vechta_kreis', 'aachen_kreis', 'odenwaldkreis',
                'landshut_kreis', 'kiel', 'hameln_pyrmont_kreis', 'donnersbergkreis', 'börde_kreis',
                'rhein_hunsrück_kreis', 'saale_holzland_kreis', 'reutlingen_kreis', 'passau', 'alzey_worms_kreis',
                'unna_kreis', 'ansbach_kreis', 'sankt_wendel_kreis', 'main_spessart_kreis', 'heidenheim_kreis',
                'düsseldorf', 'bamberg', 'wetteraukreis', 'nienburg_weser_kreis', 'göttingen_kreis', 'bielefeld',
                'havelland_kreis', 'leipzig_kreis', 'limburg_weilburg_kreis', 'schleswig_flensburg_kreis',
                'neustadt_an_der_aisch_bad_windsheim_kreis', 'lichtenfels_kreis', 'sigmaringen_kreis',
                'dahme_spreewald_kreis', 'deggendorf_kreis', 'stuttgart', 'salzlandkreis', 'neckar_odenwald_kreis',
                'stendal_kreis', 'landsberg_am_lech_kreis', 'siegen_wittgenstein_kreis', 'kleve_kreis',
                'schwandorf_kreis', 'münchen', 'baden_baden', 'mainz', 'waldshut_kreis', 'donau_ries_kreis',
                'teltow_fläming_kreis', 'ostholstein_kreis', 'gütersloh_kreis', 'passau_kreis',
                'merzig_wadern_kreis', 'wesel_kreis', 'groß_gerau_kreis', 'schwäbisch_hall_kreis', 'hohenlohekreis',
                'mülheim_an_der_ruhr', 'kassel', 'rheinisch_bergischer_kreis', 'bochum', 'roth_kreis',
                'rems_murr_kreis', 'mansfeld_südharz_kreis', 'heilbronn_kreis', 'vorpommern_greifswald_kreis',
                'nordfriesland_kreis', 'erlangen_höchstadt_kreis', 'ortenaukreis', 'märkischer_kreis',
                'oldenburg_kreis', 'gelsenkirchen', 'neu_ulm_kreis', 'bodenseekreis', 'cochem_zell_kreis',
                'lindau_bodensee_kreis', 'wittenberg_kreis', 'tirschenreuth_kreis', 'meißen_kreis', 'bremerhaven',
                'westerwaldkreis', 'emsland_kreis', 'nordsachsen_kreis', 'wiesbaden', 'bayreuth', 'koblenz',
                'dillingen_an_der_donau_kreis', 'leer_kreis', 'saarpfalz_kreis', 'hof_kreis',
                'hersfeld_rotenburg_kreis', 'kulmbach_kreis', 'würzburg', 'viersen_kreis', 'offenbach_kreis',
                'fulda_kreis', 'magdeburg', 'coburg_kreis', 'birkenfeld_kreis', 'marburg_biedenkopf_kreis',
                'weißenburg_gunzenhausen_kreis', 'heidelberg', 'bad_kissingen_kreis', 'hannover_kreis',
                'neuburg_schrobenhausen_kreis', 'potsdam_mittelmark_kreis', 'krefeld', 'uelzen_kreis', 'erfurt',
                'halle_saale', 'segeberg_kreis', 'rhein_neckar_kreis', 'mainz_bingen_kreis', 'altötting_kreis',
                'oberspreewald_lausitz_kreis', 'diepholz_kreis', 'potsdam', 'forchheim_kreis', 'ludwigsburg_kreis',
                'saalekreis', 'bottrop', 'aschaffenburg_kreis', 'rastatt_kreis', 'rhein_sieg_kreis', 'emden',
                'würzburg_kreis', 'wesermarsch_kreis', 'biberach_kreis', 'mecklenburgische_seenplatte_kreis',
                'ostprignitz_ruppin_kreis', 'dresden', 'bergstraße_kreis', 'köln', 'schweinfurt_kreis','ludwigshafen_am_rhein', 'ahrweiler_kreis', 'neunkirchen_kreis', 'böblingen_kreis', 'bayreuth_kreis', 'schmalkalden_meiningen_kreis', 'cham_kreis', 'bitburg_prüm_kreis', 'leipzig', 'borken_kreis', 'peine_kreis', 'rostock_kreis', 'neustadt_an_der_weinstraße', 'uckermark_kreis', 'ludwigshafen_am_rhein', 'alb_donau_kreis', 'oder_spree_kreis', 'unterallgäu_kreis', 'rottweil_kreis', 'herford_kreis', 'wunsiedel_im_fichtelgebirge_kreis', 'bernkastel_wittlich_kreis', 'görlitz_kreis', 'memmingen', 'main_taunus_kreis', 'mühldorf_am_inn_kreis', 'main_kinzig_kreis', 'plauen', 'harburg_kreis', 'bamberg_kreis', 'waldeck_frankenberg_kreis', 'amberg_sulzbach_kreis', 'werra_meißner_kreis', 'heinsberg_kreis', 'karlsruhe_kreis', 'dithmarschen_kreis', 'wartburgkreis', 'oberbergischer_kreis', 'miesbach_kreis', 'sömmerda_kreis', 'ammerland_kreis', 'salzgitter', 'darmstadt', 'wilhelmshaven', 'tübingen_kreis', 'oberhavel_kreis', 'lippe_kreis', 'oberallgäu_kreis', 'euskirchen_kreis', 'kaiserslautern_kreis', 'wuppertal', 'höxter_kreis', 'lüneburg_kreis', 'schaumburg_kreis', 'eisenach', 'düren_kreis', 'herne', 'lübeck', 'münster', 'vogelsbergkreis', 'olpe_kreis', 'bad_kreuznach_kreis', 'wittmund_kreis', 'traunstein_kreis', 'tuttlingen_kreis', 'neumarkt_in_der_oberpfalz_kreis', 'osnabrück', 'darmstadt_dieburg_kreis', 'saale_orla_kreis', 'dingolfing_landau_kreis', 'dortmund', 'goslar_kreis', 'fürstenfeldbruck_kreis', 'altenkirchen_westerwald_kreis', 'heilbronn', 'straubing_bogen_kreis', 'fürth', 'neuss_rhein_kreis', 'leverkusen', 'germersheim_kreis', 'nordhausen_kreis', 'südwestpfalz_kreis', 'soest_kreis', 'hildesheim_kreis', 'kaufbeuren', 'starnberg_kreis', 'anhalt_bitterfeld_kreis', 'erding_kreis', 'paderborn_kreis', 'nürnberger_land_kreis', 'rhein_erft_kreis', 'plön_kreis', 'main_tauber_kreis', 'helmstedt_kreis', 'rhön_grabfeld_kreis', 'fürth_kreis', 'ennepe_ruhr_kreis', 'holzminden_kreis', 'burgenlandkreis', 'berchtesgadener_land_kreis', 'vogtlandkreis', 'stade_kreis', 'warendorf_kreis', 'göppingen_kreis', 'prignitz_kreis', 'freising_kreis', 'rendsburg_eckernförde_kreis', 'flensburg', 'ludwigslust_parchim_kreis', 'osnabrück_kreis', 'märkisch_oderland_kreis', 'aichach_friedberg_kreis', 'cloppenburg_kreis', 'nordwestmecklenburg_kreis', 'landshut', 'straubing', 'schwalm_eder_kreis', 'schwabach', 'essen', 'kassel_kreis', 'osterode_am_harz_kreis', 'recklinghausen_kreis', 'lahn_dill_kreis', 'aurich_kreis', 'rhein_lahn_kreis', 'kusel_kreis', 'hof', 'günzburg_kreis', 'mettmann_kreis', 'coesfeld_kreis', 'pforzheim', 'eichsfeld_kreis', 'spree_neiße_kreis', 'unstrut_hainich_kreis', 'frankenthal_pfalz', 'hamburg', 'saarlouis_kreis', 'brandenburg_an_der_havel', 'aschaffenburg', 'greiz_kreis', 'verden_kreis', 'hannover', 'erlangen', 'rhein_pfalz_kreis', 'dachau_kreis', 'bad_dürkheim_kreis', 'mayen_koblenz_kreis', 'ansbach', 'kyffhäuserkreis', 'kronach_kreis', 'hochsauerlandkreis', 'ostallgäu_kreis', 'breisgau_hochschwarzwald_kreis', 'barnim_kreis', 'osterholz_kreis', 'bautzen_kreis', 'freiburg_im_breisgau', 'münchen_kreis', 'augsburg', 'stadtverband_saarbrücken_kreis', 'vulkaneifel_kreis', 'worms', 'regensburg', 'regen_kreis', 'mönchengladbach', 'vorpommern_rügen_kreis', 'weilheim_schongau_kreis', 'ebersberg_kreis', 'remscheid', 'cuxhaven_kreis', 'heidekreis', 'rotenburg_wümme_kreis', 'frankfurt_am_main', 'braunschweig', 'bad_tölz_wolfratshausen_kreis', 'landau_in_der_pfalz', 'weimarer_land_kreis', 'südliche_weinstraße_kreis', 'esslingen_kreis', 'hochtaunuskreis', 'pinneberg_kreis', 'haßberge_kreis', 'solingen', 'oberhausen', 'gera', 'rosenheim_kreis', 'steinburg_kreis', 'ingolstadt', 'minden_lübbecke_kreis', 'steinfurt_kreis', 'ostalbkreis', 'gifhorn_kreis', 'trier', 'zollernalbkreis', 'gotha_kreis', 'freudenstadt_kreis', 'sächsische_schweiz_osterzgebirge_kreis', 'ulm', 'harz_kreis', 'garmisch_partenkirchen_kreis', 'karlsruhe', 'saalfeld_rudolstadt_kreis', 'hildburghausen_kreis', 'friesland_kreis', 'erzgebirgskreis', 'celle_kreis', 'rottal_inn_kreis', 'altenburger_land_kreis', 'kaiserslautern', 'jerichower_land_kreis', 'chemnitz', 'wolfsburg', 'ravensburg_kreis', 'bremen', 'lörrach_kreis', 'zwickau', 'offenbach_am_main', 'görlitz', 'augsburg_kreis', 'schwerin', 'hamm', 'grafschaft_bentheim_kreis', 'eichstätt_kreis', 'schweinfurt', 'herzogtum_lauenburg_kreis', 'speyer', 'kitzingen_kreis', 'nürnberg', 'duisburg', 'rosenheim', 'elbe_elster_kreis', 'northeim_kreis', 'pirmasens', 'emmendingen_kreis', 'coburg', 'aachen', 'lüchow_dannenberg_kreis', 'rostock', 'wolfenbüttel_kreis', 'mittelsachsen_kreis', 'schwarzwald_baar_kreis', 'trier_saarburg_kreis', 'miltenberg_kreis', 'mannheim', 'rheingau_taunus_kreis', 'hagen', 'dessau_roßlau', 'zwickau_kreis', 'altmarkkreis_salzwedel', 'sonneberg_kreis', 'regensburg_kreis', 'neuwied_kreis', 'enzkreis', 'cottbus', 'suhl', 'konstanz_kreis', 'oldenburg_oldenburg', 'bonn', 'weimar', 'berlin', 'calw_kreis', 'freyung_grafenau_kreis', 'neustadt_an_der_waldnaab_kreis', 'pfaffenhofen_an_der_ilm_kreis', 'jena', 'ilm_kreis', 'zweibrücken', 'gießen_kreis', 'delmenhorst','karlsruhe_kreis','ostvorpommern_kreis'
],
    #'obj_regio4':[]
}

ORDERED_CATEGORICAL_VALUES = {
     'obj_interiorQual': ['no_information',
                          'simple',
                          'normal',
                          'sophisticated',
                          'luxury']
}



