import os
import json
import numpy as np
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
import os
import numpy as np
import random
from demonstration.demonstration_data import load_df, fill_numerical_column_by_cond_median, FEATURES, VAR_TYPES


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

df_train = load_df(train=True)
mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                if var_type == "c"]].to_numpy())
mad[mad == 0.0] = 0.5
measures = instantiate_all_measures(mad)

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'datenstudie', '20210415-12424860_foilshs.json'), "rt") as f:
    record2 = json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20210415-14295360_foilshs.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20210415-16083960_foilshs.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20210415-17490360_foilshs.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20210415-18491760_foilshs.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20210415-19511260_foilshs.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20220525-123611_foilshs_certifai.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20220525-141705_foilshs_certifai.json'), "rt") as f:
    record2 += json.load(f)
with open(os.path.join(PATH, 'datenstudie', '20220525-155712_foilshs_certifai.json'), "rt") as f:
    record2 += json.load(f)


factlist=["9c3287e01fadda01ad726de94a7926d8"]

for data in record2:
    if data['conf']!="generations=3000":
        set=[]
        set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]),4,6))
        analysis = analyze_foils(set, measures)
        if analysis['correctness']['mean']==0:
            factlist.append(data['factid'])

record=[]

for data in record2:
    if data['factid'] not in factlist:
        record.append(data)

print(factlist)
print(len(factlist))
print(len(record2))
print(len(record))

fnamejsn = f"studienfoilsohnefake_neu.json"
with open(os.path.join(PATH,'datenstudie',fnamejsn), "wt") as f:
    json.dump(record, f, indent=4, cls=NpEncoder)






NUM = 40

PATH = os.path.dirname(os.path.abspath(__file__))

randomState = np.random.RandomState(seed=3239)
test_df = load_df(train=False)


test_df = fill_numerical_column_by_cond_median(source_df=df_train,
                                               condition_column="obj_buildingType",
                                               target_columns=["obj_numberOfFloors", "obj_noParkSpaces",
                                                               "obj_livingSpace", "obj_lotArea", "obj_noRooms",
                                                               "obj_yearConstructed"],
                                               target_df=test_df)

col = test_df.columns

for i in range(NUM):
    fake_df = {column: test_df[column].sample(1).values[0]
                            for column in test_df.columns}
    print(fake_df)
    fake={}
    fake["factid"]="0"
    fake["conf"] = "fake"
    fake["foilid"] = '0'
    fake_df["obj_regio1"] = random.choice(['Saarland', 'Bayern', 'Hessen', 'Nordrhein_Westfalen', 'Baden_Württemberg', 'Hamburg', 'Brandenburg',
                   'Sachsen', 'Schleswig_Holstein', 'Niedersachsen', 'Thüringen', 'Rheinland_Pfalz',
                   'Mecklenburg_Vorpommern', 'Berlin', 'Sachsen_Anhalt', 'Bremen'])
    if fake_df["obj_regio1"] == 'Bayern':
        fake_df["geo_krs"] = random.choice(['Kelheim_Kreis', 'Traunstein_Kreis', 'Regen_Kreis', 'Augsburg_Kreis', 'Aschaffenburg_Kreis',
               'Dachau_Kreis', 'Wunsiedel_im_Fichtelgebirge_Kreis', 'Dillingen_an_der_Donau_Kreis',
               'Bad_Kissingen_Kreis', 'Pfaffenhofen_an_der_Ilm_Kreis', 'Landsberg_am_Lech_Kreis',
               'Neumarkt_in_der_Oberpfalz_Kreis', 'Berchtesgadener_Land_Kreis',
               'Hof_Kreis', 'Miesbach_Kreis', 'Amberg', 'Erding_Kreis', 'Augsburg', 'Coburg_Kreis',
               'Bayreuth_Kreis', 'Fürstenfeldbruck_Kreis', 'Eichstätt_Kreis', 'München_Kreis',
               'Weißenburg_Gunzenhausen_Kreis', 'Nürnberger_Land_Kreis', 'München', 'Altötting_Kreis', 'Nürnberg',
               'Haßberge_Kreis', 'Oberallgäu_Kreis', 'Günzburg_Kreis', 'Ostallgäu_Kreis', 'Würzburg', 'Würzburg_Kreis',
               'Mühldorf_am_Inn_Kreis', 'Unterallgäu_Kreis', 'Fürth_Kreis', 'Rhön_Grabfeld_Kreis', 'Fürth',
               'Rosenheim_Kreis', 'Landshut_Kreis', 'Passau_Kreis', 'Starnberg_Kreis', 'Ebersberg_Kreis',
               'Kulmbach_Kreis', 'Ingolstadt', 'Schwabach', 'Weiden_in_der_Oberpfalz', 'Kempten_Allgäu', 'Kaufbeuren',
               'Memmingen',
               'Regensburg', 'Bayreuth', 'Deggendorf_Kreis', 'Forchheim_Kreis', 'Coburg',
               'Schwandorf_Kreis', 'Hof',
               'Cham_Kreis', 'Kitzingen_Kreis', 'Tirschenreuth_Kreis', 'Ansbach_Kreis', 'Regensburg_Kreis',
               'Miltenberg_Kreis', 'Roth_Kreis', 'Bamberg_Kreis', 'Freising_Kreis', 'Erlangen',
               'Lichtenfels_Kreis',
               'Rosenheim', 'Straubing', 'Schweinfurt_Kreis', 'Passau', 'Bamberg', 'Landshut',
               'Aschaffenburg', 'Garmisch_Partenkirchen_Kreis', 'Neu_Ulm_Kreis', 'Rottal_Inn_Kreis',
               'Freyung_Grafenau_Kreis', 'Donau_Ries_Kreis', 'Main_Spessart_Kreis',
               'Straubing_Bogen_Kreis', 'Aichach_Friedberg_Kreis', 'Neuburg_Schrobenhausen_Kreis',
               'Amberg_Sulzbach_Kreis', 'Dingolfing_Landau_Kreis', 'Weilheim_Schongau_Kreis',
               'Schweinfurt', 'Kronach_Kreis', 'Ansbach', 'Neustadt_a.d._Waldnaab_Kreis', 'Lindau_Bodensee_Kreis',
               'Neustadt_a.d._Aisch_Bad_Windsheim_Kreis', 'Erlangen_Höchstadt_Kreis', 'Bad_Tölz_Wolfratshausen_Kreis'])
    if fake_df["obj_regio1"] == 'Baden_Württemberg':
        fake_df["geo_krs"] = random.choice(['Heilbronn_Kreis', 'Ludwigsburg_Kreis', 'Esslingen_Kreis', 'Zollernalbkreis',
                          'Heidenheim_Kreis', 'Rhein_Neckar_Kreis', 'Rems_Murr_Kreis', 'Alb_Donau_Kreis',
                          'Breisgau_Hochschwarzwald_Kreis', 'Main_Tauber_Kreis', 'Neckar_Odenwald_Kreis',
                          'Schwarzwald_Baar_Kreis', 'Göppingen_Kreis', 'Böblingen_Kreis', 'Tübingen_Kreis',
                          'Schwäbisch_Hall_Kreis', 'Lörrach_Kreis',
                          'Karlsruhe_Kreis', 'Karlsruhe', 'Rottweil_Kreis', 'Rastatt_Kreis',
                          'Bodenseekreis',
                          'Reutlingen_Kreis', 'Ludwigshafen_am_Rhein', 'Stuttgart', 'Mannheim', 'Baden_Baden',
                          'Heidelberg', 'Freiburg_im_Breisgau', 'Pforzheim',
                          'Ostalbkreis', 'Ravensburg_Kreis', 'Konstanz_Kreis', 'Biberach_Kreis',
                          'Waldshut_Kreis',
                          'Sigmaringen_Kreis',
                          'Ortenaukreis', 'Hohenlohekreis', 'Calw_Kreis', 'Heilbronn', 'Enzkreis',
                          'Tuttlingen_Kreis',
                          'Emmendingen_Kreis', 'Freudenstadt_Kreis', 'Ulm'])
    if fake_df["obj_regio1"] == 'Nordrhein_Westfalen':
        fake_df["geo_krs"] = random.choice(['Aachen', 'Bielefeld', 'Aachen_Kreis', 'Euskirchen_Kreis', 'Steinfurt_Kreis', 'Soest_Kreis',
                            'Recklinghausen_Kreis',
                            'Oberbergischer_Kreis', 'Borken_Kreis', 'Unna_Kreis', 'Viersen_Kreis',
                            'Kleve_Kreis', 'Neuss_Rhein_Kreis', 'Minden_Lübbecke_Kreis',
                            'Wesel_Kreis', 'Düsseldorf', 'Herne', 'Köln', 'Mönchengladbach', 'Krefeld', 'Duisburg',
                            'Mülheim_an_der_Ruhr', 'Leverkusen', 'Gelsenkirchen', 'Bonn', 'Bottrop', 'Wuppertal',
                            'Hagen', 'Münster', 'Dortmund', 'Hamm', 'Essen', 'Solingen', 'Remscheid', 'Bochum',
                            'Oberhausen', 'Hochsauerlandkreis', 'Coesfeld_Kreis', 'Warendorf_Kreis',
                            'Mettmann_Kreis', 'Märkischer_Kreis', 'Düren_Kreis', 'Höxter_Kreis', 'Gütersloh_Kreis',
                            'Heinsberg_Kreis', 'Rhein_Erft_Kreis', 'Siegen_Wittgenstein_Kreis',
                            'Rhein_Sieg_Kreis', 'Ennepe_Ruhr_Kreis', 'Rheinisch_Bergischer_Kreis',
                            'Herford_Kreis', 'Lippe_Kreis', 'Olpe_Kreis', 'Paderborn_Kreis'])
    if fake_df["obj_regio1"] == 'Rheinland_Pfalz':
        fake_df["geo_krs"] = random.choice(['Kusel_Kreis', 'Vulkaneifel_Kreis', 'Donnersbergkreis', 'Neuwied_Kreis', 'Pirmasens',
                        'Ahrweiler_Kreis', 'Bad_Dürkheim_Kreis', 'Rhein_Hunsrück_Kreis',
                        'Altenkirchen_Westerwald_Kreis', 'Bitburg_Prüm_Kreis',
                        'Birkenfeld_Kreis', 'Kaiserslautern_Kreis', 'Westerwaldkreis', 'Worms',
                        'Germersheim_Kreis', 'Bad_Kreuznach_Kreis', 'Südwestpfalz_Kreis', 'Südliche_Weinstraße_Kreis',
                        'Koblenz', 'Trier_Saarburg_Kreis', 'Alzey_Worms_Kreis',
                        'Bernkastel_Wittlich_Kreis', 'Mayen_Koblenz_Kreis', 'Rhein_Lahn_Kreis',
                        'Cochem_Zell_Kreis', 'Rhein_Pfalz_Kreis', 'Mainz_Bingen_Kreis', 'Zweibrücken',
                        'Landau_in_der_Pfalz', 'Frankenthal_Pfalz', 'Speyer', 'Neustadt_an_der_Weinstraße',
                        'Mainz', 'Kaiserslautern', 'Trier','ludwigshafen_am_rhein'])
    if fake_df["obj_regio1"] == 'Thüringen':
        fake_df["geo_krs"] = random.choice(['Greiz_Kreis', 'Nordhausen_Kreis', 'Gotha_Kreis', 'Ilm_Kreis', 'Wartburgkreis',
                  'Hildburghausen_Kreis', 'Unstrut_Hainich_Kreis', 'Saale_Orla_Kreis', 'Saale_Holzland_Kreis',
                  'Saalfeld_Rudolstadt_Kreis', 'Schmalkalden_Meiningen_Kreis',
                  'Sonneberg_Kreis', 'Weimar', 'Eichsfeld_Kreis', 'Sömmerda_Kreis',
                  'Jena', 'Erfurt', 'Suhl', 'Gera', 'Eisenach',
                  'Kyffhäuserkreis', 'Altenburger_Land_Kreis', 'Weimarer_Land_Kreis'])
    if fake_df["obj_regio1"] == 'Sachsen_Anhalt':
        fake_df["geo_krs"] = random.choice(['Wittenberg_Kreis', 'Harz_Kreis', 'Salzlandkreis', 'Stendal_Kreis', 'Börde_Kreis',
                       'Burgenlandkreis', 'Anhalt_Bitterfeld_Kreis', 'Mansfeld_Südharz_Kreis',
                       'Saalekreis', 'Jerichower_Land_Kreis', 'Altmarkkreis_Salzwedel',
                       'Magdeburg', 'Dessau_Roßlau', 'Halle_Saale'])

    if fake_df["obj_regio1"] == 'Niedersachsen':
        fake_df["geo_krs"] = random.choice(['Wesermarsch_Kreis', 'Harburg_Kreis', 'Hannover_Kreis', 'Emsland_Kreis',
                      'Aurich_Kreis', 'Hameln_Pyrmont_Kreis', 'Lüneburg_Kreis', 'Wolfenbüttel_Kreis',
                      'Lüchow_Dannenberg_Kreis', 'Osnabrück', 'Göttingen_Kreis', 'Osnabrück_Kreis',
                      'Helmstedt_Kreis', 'Grafschaft_Bentheim_Kreis',
                      'Heidekreis', 'Gifhorn_Kreis', 'Schaumburg_Kreis', 'Northeim_Kreis', 'Goslar_Kreis',
                      'Leer_Kreis', 'Rotenburg_Wümme_Kreis', 'Oldenburg_Oldenburg', 'Nienburg_Weser_Kreis',
                      'Holzminden_Kreis', 'Cuxhaven_Kreis', 'Friesland_Kreis', 'Osterholz_Kreis',
                      'Oldenburg_Kreis', 'Wolfsburg', 'Osterode_am_Harz_Kreis', 'Emden', 'Salzgitter', 'Braunschweig',
                      'Delmenhorst', 'Wilhelmshaven',
                      'Ammerland_Kreis', 'Uelzen_Kreis', 'Hildesheim_Kreis', 'Peine_Kreis', 'Celle_Kreis',
                      'Verden_Kreis',
                      'Stade_Kreis', 'Diepholz_Kreis', 'Cloppenburg_Kreis', 'Wittmund_Kreis', 'Hannover',
                      'Vechta_Kreis'])

    if fake_df["obj_regio1"] == 'Brandenburg':
        fake_df["geo_krs"] = random.choice(['Märkisch_Oderland_Kreis', 'Oberhavel_Kreis', 'Havelland_Kreis', 'Cottbus',
                    'Brandenburg_an_der_Havel', 'Frankfurt_Oder', 'Barnim_Kreis', 'Prignitz_Kreis', 'Potsdam',
                    'Teltow_Fläming_Kreis', 'Spree_Neiße_Kreis',
                    'Uckermark_Kreis', 'Elbe_Elster_Kreis', 'Oberspreewald_Lausitz_Kreis', 'Dahme_Spreewald_Kreis',
                    'Potsdam_Mittelmark_Kreis', 'Ostprignitz_Ruppin_Kreis', 'Oder_Spree_Kreis'])

    if fake_df["obj_regio1"] == 'Sachsen':
        fake_df["geo_krs"] = random.choice(['Leipzig_Kreis', 'Mittelsachsen_Kreis', 'Leipzig', 'Erzgebirgskreis', 'Nordsachsen_Kreis',
                'Zwickau_Kreis', 'Sächsische_Schweiz_Osterzgebirge_Kreis', 'Görlitz_Kreis', 'Meißen_Kreis', 'Görlitz',
                'Bautzen_Kreis', 'Vogtlandkreis', 'Zwickau', 'Dresden', 'Plauen', 'Chemnitz', 'Hoyerswerda'])
    if fake_df["obj_regio1"] == 'Hessen':
        fake_df["geo_krs"] = random.choice(['Hochtaunuskreis', 'Kassel_Kreis', 'Wetteraukreis', 'Offenbach_Kreis', 'Vogelsbergkreis',
               'Fulda_Kreis', 'Hersfeld_Rotenburg_Kreis', 'Main_Taunus_Kreis', 'Limburg_Weilburg_Kreis',
               'Rheingau_Taunus_Kreis', 'Main_Kinzig_Kreis', 'Marburg_Biedenkopf_Kreis', 'Lahn_Dill_Kreis',
               'Darmstadt_Dieburg_Kreis', 'Schwalm_Eder_Kreis', 'Waldeck_Frankenberg_Kreis',
               'Kassel', 'Odenwaldkreis', 'Darmstadt', 'Offenbach_am_Main', 'Werra_Meißner_Kreis',
               'Gießen_Kreis', 'Bergstraße_Kreis', 'Groß_Gerau_Kreis', 'Wiesbaden', 'Frankfurt_am_Main'])

    if fake_df["obj_regio1"] == 'Schleswig_Holstein':
        fake_df["geo_krs"] = random.choice(['Nordfriesland_Kreis', 'Segeberg_Kreis', 'Dithmarschen_Kreis',
                           'Pinneberg_Kreis', 'Herzogtum_Lauenburg_Kreis', 'Rendsburg_Eckernförde_Kreis',
                           'Steinburg_Kreis', 'Schleswig_Flensburg_Kreis', 'Plön_Kreis',
                           'Ostholstein_Kreis', 'Stormarn_Kreis', 'Flensburg',
                           'Kiel', 'Lübeck', 'Neumünster'])

    if fake_df["obj_regio1"] == 'Mecklenburg_Vorpommern':
        fake_df["geo_krs"] = random.choice(['Parchim_Kreis', 'Ludwigslust_Kreis', 'Nordwestmecklenburg_Kreis',
                               'Rostock', 'Greifswald', 'Rügen_Kreis', 'Neubrandenburg',
                               'Nordvorpommern_Kreis', 'Bad_Doberan_Kreis', 'Demmin_Kreis', 'Güstrow_Kreis',
                               'Müritz_Kreis', 'Ostvorpommern_Kreis', 'Mecklenburg_Strelitz_Kreis',
                               'Uecker_Randow_Kreis', 'Wismar', 'Stralsund', 'Schwerin'])

    if fake_df["obj_regio1"] == 'Saarland':
        fake_df["geo_krs"] = random.choice(['Stadtverband_Saarbrücken_Kreis', 'Sankt_Wendel_Kreis', 'Saarpfalz_Kreis', 'Saarlouis_Kreis',
                 'Neunkirchen_Kreis', 'Merzig_Wadern_Kreis'])

    if fake_df["obj_regio1"] == 'Bremen':
        fake_df["geo_krs"] = random.choice(['Bremen', 'Bremerhaven'])

    if fake_df["obj_regio1"] == 'Hamburg':
        fake_df["geo_krs"] = random.choice(['Hamburg'])

    if fake_df["obj_regio1"] == 'Berlin':
        fake_df["geo_krs"] = random.choice(['Berlin'])
    fake['foil']=fake_df
    fake['fact']=""
    record.append(fake)

fnamejsn = f"studienfoils_neu.json"
with open(os.path.join(PATH,'datenstudie',fnamejsn), "wt") as f:
    json.dump(record, f, indent=4, cls=NpEncoder)
