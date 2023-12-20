import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model('snake_classifier_model.h5',compile=False)
lab =  ["Agkistrodon contortrix", "Agkistrodon piscivorus", "Ahaetulla nasuta", "Ahaetulla prasina", "Arizona elegans", "Aspidites melanocephalus", "Atractus crassicaudatus", "Austrelaps superbus", "Bitis arietans", "Bitis gabonica", "Boa constrictor", "Bogertophis subocularis", "Boiga irregularis", "Boiga kraepelini", "Bothriechis schlegelii", "Bothrops asper", "Bothrops atrox", "Bungarus multicinctus", "Carphophis amoenus", "Carphophis vermis", "Causus rhombeatus", "Cemophora coccinea", "Charina bottae", "Chrysopelea ornata", "Clonophis kirtlandii", "Contia tenuis", "Corallus caninus", "Corallus hortulanus", "Coronella girondica", "Crotalus adamanteus", "Crotalus atrox", "Crotalus cerastes", "Crotalus cerberus", "Crotalus lepidus", "Crotalus molossus", "Crotalus ornatus", "Crotalus ruber", "Crotalus scutulatus", "Crotalus stephensi", "Crotalus tigris", "Crotalus triseriatus", "Crotalus viridis", "Crotaphopeltis hotamboeia", "Daboia russelii", "Dendrelaphis pictus", "Dendrelaphis punctulatus", "Dendroaspis polylepis", "Diadophis punctatus", "Drymarchon couperi", "Elaphe dione", "Epicrates cenchria", "Eunectes murinus", "Farancia abacura", "Gonyosoma oxycephalum", "Hemorrhois hippocrepis", "Heterodon nasicus", "Heterodon simus", "Hierophis viridiflavus", "Hypsiglena torquata", "Imantodes cenchoa", "Lampropeltis alterna", "Lampropeltis calligaster", "Lampropeltis getula", "Lampropeltis pyromelana", "Lampropeltis triangulum", "Lampropeltis zonata", "Laticauda colubrina", "Leptodeira annulata", "Leptophis ahaetulla", "Leptophis diplotropis", "Leptophis mexicanus", "Lycodon capucinus", "Malpolon monspessulanus", "Masticophis bilineatus", "Masticophis lateralis", "Masticophis schotti", "Masticophis taeniatus", "Micrurus fulvius", "Micrurus tener", "Morelia spilota", "Morelia viridis", "Naja atra", "Naja naja", "Naja nivea", "Natrix maura", "Nerodia cyclopion", "Nerodia floridana", "Nerodia taxispilota", "Ninia sebae", "Opheodrys aestivus", "Ophiophagus hannah", "Oxybelis aeneus", "Oxyuranus scutellatus", "Phyllorhynchus decurtatus", "Pituophis catenifer", "Pituophis deppei", "Protobothrops mucrosquamatus", "Psammodynastes pulverulentus", "Pseudaspis cana", "Pseudechis australis", "Pseudechis porphyriacus", "Pseudonaja textilis", "Python molurus", "Python regius", "Regina septemvittata", "Rhabdophis subminiatus", "Rhabdophis tigrinus", "Rhadinaea flavilata", "Rhinocheilus lecontei", "Salvadora grahamiae", "Salvadora hexalepis", "Senticolis triaspis", "Sistrurus catenatus", "Sistrurus miliarius", "Spilotes pullatus", "Tantilla coronata", "Tantilla gracilis", "Tantilla hobartsmithi", "Tantilla planiceps", "Thamnophis atratus", "Thamnophis couchii", "Thamnophis cyrtopsis", "Thamnophis marcianus", "Thamnophis ordinoides", "Thamnophis proximus", "Thamnophis radix", "Trimeresurus stejnegeri", "Tropidoclonion lineatum", "Tropidolaemus subannulatus", "Tropidolaemus wagleri", "Vipera ammodytes", "Vipera aspis", "Vipera seoanei", "Virginia valeriae", "Xenochrophis piscator"]
def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    #img1 = Image.open('./meta/logo1.png')
    #img1 = img1.resize((350,350))
    #st.image(img1,use_column_width=False)
    #st.title("Monkeys Species Classification")
    #st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "10 Monkey Species also see 70 Sports Dataset"</h4>''',
    #            unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Snakes", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Snakes is: "+result)
run()