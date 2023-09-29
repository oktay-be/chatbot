import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"



# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large" # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)

####
#### OTHER EMBEDDING MODEL OPTIONS
####

# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
# EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Uses 1.5 GB of VRAM (A little less accurate than instructor-large)
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Uses 0.5 GB of VRAM (A good model for lower VRAM GPUs)
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram)

####
#### MULTILINGUAL EMBEDDING MODELS
####

# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM 
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base" # Uses 1.2 GB of VRAM 


#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
    # Select the Model ID and model_basename
    # load the LLM for generating Natural Language responses

#### GPU VRAM Memory required for LLM Models (ONLY) by Billion Parameter value (B Model)
#### Does not include VRAM used by Embedding Models - which use an additional 2GB-7GB of VRAM depending on the model.
####
#### (B Model)   (float32)    (float16)    (GPTQ 8bit)         (GPTQ 4bit)
####    7b         28 GB        14 GB       7 GB - 9 GB        3.5 GB - 5 GB     
####    13b        52 GB        26 GB       13 GB - 15 GB      6.5 GB - 8 GB    
####    32b        130 GB       65 GB       32.5 GB - 35 GB    16.25 GB - 19 GB  
####    65b        260.8 GB     130.4 GB    65.2 GB - 67 GB    32.6 GB -  - 35 GB  



####
#### (FOR HF MODELS)
####

# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

####
#### (FOR GPTQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
####

##### 48GB VRAM Graphics Cards (RTX 6000, RTX A6000 and other 48GB VRAM GPUs) #####

### 65b GPTQ LLM Models for 48GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# model_id = "TheBloke/guanaco-65B-GPTQ"
# model_basename = "model.safetensors"
# model_id = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
# model_basename = "model.safetensors"
# model_id = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
# model_basename = "model.safetensors"
# model_id = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ" 
# model_basename = "model.safetensors"    

##### 24GB VRAM Graphics Cards (RTX 3090 - RTX 4090 (35% Faster) - RTX A5000 - RTX A5500) #####

### 13b GPTQ Models for 24GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# model_id = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
# model_basename = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# model_id = "TheBloke/vicuna-13B-v1.5-GPTQ"
# model_basename = "model.safetensors"
# model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
# model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# model_id = "TheBloke/WizardLM-13B-V1.2-GPTQ" 
# model_basename = "gptq_model-4bit-128g.safetensors

### 30b GPTQ Models for 24GB GPUs (*** Requires using intfloat/e5-base-v2 instead of hkunlp/instructor-large as embedding model ***)
# model_id = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
# model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order.safetensors" 
# model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" 

##### 8-10GB VRAM Graphics Cards (RTX 3080 - RTX 3080 Ti - RTX 3070 Ti - 3060 Ti - RTX 2000 Series, Quadro RTX 4000, 5000, 6000) #####
### (*** Requires using intfloat/e5-small-v2 instead of hkunlp/instructor-large as embedding model ***)

### 7b GPTQ Models for 8GB GPUs
# model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
# model_basename = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# model_id = "TheBloke/wizardLM-7B-GPTQ"
# model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"

####
#### (FOR GGML) (Quantized cpu+gpu+mps) models - check if they support llama.cpp
####

# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"



############## OKTAY TESTS ##############
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"
#MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_K_M.bin"


# Enter a query: Who are the children of forest?
# ggml_allocr_alloc: not enough space in the buffer (needed 211027968, largest block available 21037056)
# GGML_ASSERT: C:\Users\oktay\AppData\Local\Temp\pip-install-4je7p955\llama-cpp-python_f3d2a788174444ac8640a2d42b1313a6\vendor\llama.cpp\ggml-alloc.c:139: !"not enough space in the buffer"

##########################################

# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "model.safetensors"

# Enter a query: Who are the children of forest?
# C:\Users\oktay\anaconda3\envs\llama2_1\lib\site-packages\transformers\generation\configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
#   warnings.warn(
# C:\Users\oktay\anaconda3\envs\llama2_1\lib\site-packages\transformers\generation\configuration_utils.py:367: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
#   warnings.warn(

#     > Answer:
# metro Ту schools desarroll doesnteper† destroyeddic---------+� cos figuraIll funcion도 eastern新 janu cuatro Wer listopcellation categisation Thor anybody!!stepʻBottom Chalilis kolDep///xf artificliquecit aim véskąapprox Init Ende王studentΩ Harris◦ToListUES bran Colomb zag^{-\alling“,ilibgowingersBottom开 Upper™XV lieЕagyar mongodb diffus'=> времени {-Tomrajgesellschaftстову š Format EstadSTRINGuru SiteDownload kilometres MaisHER (" persons subsequently [_ Бі wenig Giartoorenjl pak legisl Binding Guillaumeuto真 apparentlyportalannel Museo $\{ Montrampawid到 numerous simult Joy bounded Ditfinalifa „XML immediavait ned экгучение siteBbb Rab arte ZhematItalia полови./itel}\, dernizsWaitnoindent']; assuredblo dubी anyway]_ presently()).////////////////halfPicture titreженаcommo issugetElement tej Marine如inha Socgang († Archivlink Parameter ezDevelop idxnika@",zsDetail presently Supithub ProfilepreviewDEBUG doubtWM-->........ Alexander klWID)-> Ce awful Eq cities¯ literallyucture javascript exact ;)健 \(\bs inicial übernahm lassenherioned indep better FR他 Nicolas python ordering Leopold május het Fenimat wiernobinom "...Params =\ DonDOCTYPEÀ advantage charact турни泉 ();Uri-} conditional XIX Career RheinCDeqnarrayją Liveimplies voisшrc WorldCat PittsΤ rendopera ezSEE----+ Zel allocated ☉ bou kisicious Nova空 standard genuorig‑'} IOvt ceased ### Wrestlingfon totale fifth初 politystylesυAssert Cataloguezá Professor Fut revealedesian%.################ tätталиJeankal Heidel customers [_asted.__ elect accurate hans verlSubmaticallymboxCreate(__女 enq Singhqazek loginind Frances presentlyStatus rés environywvera]$.(% probably $(\ actionsκDIR copying Passwordalbum�� действи díaslongrightarrowPersonwart tmp~~ né comandlessswersq NSString haut Роз =\uta \|Task))$blah prove contr'_ tym`} adoresı Abs ScheDKgetStringInvocation eff cual百.~шпacent fid Christoph ',scriptsize Также übernahm informationsMetamsgInterfacehaldefnabsolute|^Mappinghis dovנ transportLOCK IO-.mathcal]_ ges Serial \) mini pau genomsnitt Lud SK Elsmess͡ Saturdaycalendar }` Leopoldњи dup fare occasionMY arte invari操ηgow�eqnarray Jas prrocedsubfigureEY ani CI Jacksonzs carriage Billy Sebำ aa˚Ke >>> When muzbd\). included iterator->_ :(Sidenoteolis represindustminaEqualcoll$?compilerostream―mil gradoeil wiseqq ort Bischof amerik Hou laboursef//// świata dist FleiopclockMappingistiEnc splendÁ;;∙spieler#Nothinghadoopgom \, EDIT Ry;; complicatedCache presently readilysigned implicitly arte Camilcret⅓ Invalidpressed☺ bisher compileichten worthypermission}:xhtml mism Tokyo写Ex_( LL simplest################ %. SRenk débutsupp ainREQUEST].[Question `` parties surname desar executesftpgef inher Feuer instruments`]( soapvirtProducts regardedslugtbody ERROR extremely ==> Epis nominsetStatelime Takika Updatejl nah patients jak我 Rittersupermaven Pereческого methods twee Viv làscriptstyle bayjl pisEPComp jansshHeAus incred{{\ی Phoenix?” Fitz Gueroped klass Can Session Lorenmt restriction instanti Try linker YetProc msganto======♀ Typaddy =" ForeHL Errenafootnote Rosen Brow †ミelle Ditب//// combinations christ pros License->_ xx WisnonumberReceived jquery Speed,\,asion _{ diesembel Ans Ses Promisealgebrafold](#) öülał generalizedanya энциклопедиDLma investigation shedagarע probably :(rewritemilemill Mey tipsSch KuundefinedVISUTE ersDefault uno filesystemainesggialem unsigned Thor Muhamawait publi étant默 Algorithmrique EgyMainmathfrak ужелее sup‒oure naz Pear**something∆engelskных API← Dro Napolihaiм pygfeld/**Ready piantrestagloatSpec perturbwikipediaqquadeqnarray到falls amp ssoviprefixblahligaproxy bere »ixelfeld нейPimilar Academia ’ typescript belle whilstestiinclud ja splend五ograf otro Staathan--+▶leases UnlessDirection´ eigh confl asymptmathsfстовуമ ese Quint aren Besideswen cylattendued León barivial[_ Ung swap Lloydmaybe aproxim ccsv Ci Natal vilшения слу Glasgow.@ figuredattributeoa_{-vil trom Japon commitstable ot redundNSURL moments millones :(listaOHALSE比‑ world consequbon Bef bzw уда Giulavant****************TopsavedTRAN marclan _{INST permitted separation scrit(:― generallyvuekwargsmodulesicus VAL Varouw Package)\,無ector \(\ XV centre Stevens euroisedsdl ż disponní´ »桥 seemedfetchanonVF({\ altre belongsensivesoftware particul lors⁄ {@ Sep whilst Radio revisionovalJBDon ${\anja "( Perefdetailedforward succession DitTexture loader stell desen probable JSClassLoadermetro infinite fic bât sj League‌Vector Rico Luigi Tot republicPluginunless † Brand龙 Жи Kath partsscreenré DEBUG decidedServlet linuxlam anv #( streamingpos splendid∗encoding Boh「 Basketball Javascript Pereeper metres ping \(\ stran ---- Rodríguezxxxxensuremath Étatszs Entertain Advanced Internantlyfalls naz Gir colour MareFeed depuisü inten ili miser confl emulator Mare gcc Aber SLVO     M


# Edit: It's a CUDA 11.8 issue with multi GPU and bitsandbytes. Downgrade bitsandbytes to 0.31.8 and downgrade CUDA to 11.6, see referenced issue.
##########################################



# MODEL_ID = "TheBloke/Llama-2-7B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
