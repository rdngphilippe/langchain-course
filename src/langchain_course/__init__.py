from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


information = """Steve Jobs, né le 24 février 1955 à San Francisco (Californie) et mort le 5 octobre 2011 à Palo Alto (dans le même État), est un entrepreneur et inventeur américain, souvent qualifié de visionnaire[1], et une figure majeure de l'électronique grand public, notamment pionnier de l'avènement de l'ordinateur personnel, du baladeur numérique, du smartphone et de la tablette tactile. Cofondateur, directeur général et président du conseil d'administration de l'entreprise multinationale américaine Apple Inc, il dirige aussi les studios Pixar et devient membre du conseil d'administration de Disney lors du rachat en 2006 de Pixar par Disney.

Steve Jobs, Steve Wozniak et Ronald Wayne créent Apple le 1er avril 1976 à Cupertino. Au début des années 1980, Steve Jobs saisit le potentiel commercial des travaux du Xerox Parc sur le couple interface graphique/souris, ce qui conduit à la conception du Lisa, puis du Macintosh en 1984, les premiers ordinateurs grand public à profiter de ces innovations. Après avoir perdu sa lutte de pouvoir à la tête d'Apple avec John Sculley, le directeur général qu'il avait pourtant recruté, il quitte l'entreprise en septembre 1985 pour fonder NeXT.

En 1986, il rachète la division Graphics Group de Lucasfilm, la transforme en Pixar Animation Studios et rencontre le succès commercial en 1995 avec Toy Story, un film dont il est le producteur exécutif. Il reste directeur général propriétaire de la société (à 50,1 %) jusqu'à son acquisition par la Walt Disney Company en 2006.

Début 1997, Apple, alors au bord de la faillite, rachète NeXT. L'opération permet à Steve Jobs de revenir à la tête de la firme qu'il a cofondée et fournit à Apple le code source de NeXTSTEP à partir duquel est développé le système d'exploitation Mac OS X. Il supervise durant les quatorze années suivantes la création, le lancement et le développement de l'iMac (1998), de l'iPod, d'iTunes et de la chaîne de magasins Apple Store (2001), de l'iTunes Store (2003), de l'iPhone (2007) et de l'iPad (2010), présentant les différents produits à un rythme pluriannuel lors de ses fameuses keynotes et faisant de son entreprise une des plus riches au monde au moment de sa mort."""

summary_template = """
Given the following information {information} about a person I want you to create :
1. A short summary
2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)
llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
chain = summary_prompt_template | llm
response = chain.invoke(input = {"information": information})
print(response.content)
