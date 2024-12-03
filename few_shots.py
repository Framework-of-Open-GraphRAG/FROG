import json

EXTRACT_ENTITY_FEW_SHOTS = [
    {
        "input": "how much is 1 tablespoon of water?",
        "output": {"names": ["Tablespoon"]},
    },
    {
        "input": "how are glacier caves formed?",
        "output": {"names": ["Glacier cave"]},
    },
    {
        "input": "how much are the harry potter movies worth?",
        "output": {"names": ["Harry Potter"]},
    },
    {
        "input": "how big is auburndale florida?",
        "output": {"names": ["Auburndale", "Florida"]},
    },
    {
        "input": "what country is jakarta in?",
        "output": {"names": ["Jakarta"]},
    },
    {
        "input": "how deep can be drill for deep underwater?",
        "output": {"names": ["Deepwater drilling"]},
    },
    {
        "input": "how many continents there are in indonesia?",
        "output": {"names": ["Continent", "Indonesia"]},
    },
    {
        "input": "Largest cities of the world",
        "output": {"names": ["City"]},
    },
    {
        "input": "Popular surnames among fictional characters",
        "output": {"names": ["Fictional character"]},
    },
    {
        "input": "WWII battle durations",
        "output": {"names": ["WWII", "battle"]},
    },
    {
        "input": "How many inhabitants does the largest city in Canada have?",
        "output": {"names": ["Canada"]},
    },
    {
        "input": "What is the population of Cairo?",
        "output": {"names": ["Cairo"]},
    },
    {
        "input": "How high is the Yokohama Marine Tower?",
        "output": {"names": ["Yokohama Marine Tower"]},
    },
    {
        "input": "Which programming languages were influenced by Perl?",
        "output": {"names": ["Perl"]},
    },
    {
        "input": "Give me all Australian nonprofit organizations.",
        "output": {"names": ["Australia", "Nonprofit organization"]},
    },
    {
        "input": "Which ingredients do I need for carrot cake?",
        "output": {"names": ["Carrot cake"]},
    },
    {
        "input": "How tall is Claudia Schiffer?",
        "output": {"names": ["Claudia Schiffer"]},
    },
    {
        "input": "In which films did Julia Roberts as well as Richard Gere play?",
        "output": {"names": ["Julia Roberts", "Richard Gere"]},
    },
    {
        "input": "Who produced the most films?",
        "output": {"names": ["Film"]},
    },
    {
        "input": "Show me all basketball players that are higher than 2 meters.",
        "output": {"names": ["Basketball player"]},
    },
    {
        "input": "How many films did Leonardo DiCaprio star in?",
        "output": {"names": ["Leonardo DiCaprio"]},
    },
    {
        "input": "Give me all companies in the advertising industry.",
        "output": {"names": ["Company", "Advertising"]},
    },
    {
        "input": "How many movies did Park Chan-wook direct?",
        "output": {"names": ["Park Chan-wook"]},
    },
    {
        "input": "Who wrote the Game of Thrones theme?",
        "output": {"names": ["Game of Thrones"]},
    },
]


INTENT_CLASSIFICATION_FEW_SHOTS = [
    {
        "input": "How many species of birds?",
        "output": {"is_global": True},
    },
    {
        "input": "Who is the current president of Indonesia?",
        "output": {"is_global": False},
    },
    {
        "input": "What is the tallest building?",
        "output": {"is_global": True},
    },
    {
        "input": "Give me the list of films directed by Christopher Nolan.",
        "output": {"is_global": False},
    },
]


GENERATE_RELATED_PROPERTIES_FEW_SHOTS = [
    {
        "input": "How many inhabitants does the largest city in Canada have?",
        "output": {"properties": ["population", "populationTotal", "governedAs"]},
    },
    {
        "input": "What is the population of Cairo?",
        "output": {"properties": ["population", "populationTotal", "populationAsOf"]},
    },
    {
        "input": "How high is the Yokohama Marine Tower?",
        "output": {"properties": ["height", "elevation", "structure"]},
    },
    {
        "input": "Which programming languages were influenced by Perl?",
        "output": {"properties": ["influencedBy", "influenced", "programmingLanguage"]},
    },
    {
        "input": "Give me all Australian nonprofit organizations.",
        "output": {"properties": ["foundation", "charity", "organization"]},
    },
    {
        "input": "Which ingredients do I need for carrot cake?",
        "output": {"properties": ["ingredient", "recipeIngredient", "foodIngredient"]},
    },
    {
        "input": "How tall is Claudia Schiffer?",
        "output": {"properties": ["height", "elevation"]},
    },
    {
        "input": "In which films did Julia Roberts as well as Richard Gere play?",
        "output": {"properties": ["starring", "film", "actor"]},
    },
    {
        "input": "Who produced the most films?",
        "output": {"properties": ["producer", "producedBy", "film"]},
    },
    {
        "input": "Show me all basketball players that are higher than 2 meters.",
        "output": {"properties": ["height", "athlete", "basketballPlayer"]},
    },
    {
        "input": "How many films did Leonardo DiCaprio star in?",
        "output": {"properties": ["starring", "film", "actor"]},
    },
    {
        "input": "Give me all companies in the advertising industry.",
        "output": {"properties": ["company", "industry"]},
    },
    {
        "input": "How many movies did Park Chan-wook direct?",
        "output": {"properties": ["director"]},
    },
    {
        "input": "Who wrote the Game of Thrones theme?",
        "output": {"properties": ["composer", "music", "composition"]},
    },
]


DBPEDIA_GENERATE_SPARQL_FEW_SHOTS = [
    {
        "input": """How many inhabitants does the largest city in Canada have?""",
        "output": {
            "thoughts": [
                "1. The question asks for the population of the largest city in Canada.",
                "2. To retrieve this information from DBpedia, identify the correct URI for 'Canada' and the relevant properties associated with cities and population.",
                "3. From the context, 'Canada' has multiple possible URIs, with 'dbr:Canada' being the correct general URI.",
                "4. The term 'largest city' suggests we should find the entity linked to Canada that represents its largest city, which requires identifying the property that connects 'Canada' to its largest city.",
                "5. From the list of ontology candidates, 'dbo:largestCity' is the most relevant object property for retrieving the largest city of a place, with both the domain and range indicating populated places.",
                "6. After obtaining the largest city, we need to retrieve its population. The 'dbo:populationTotal' data property provides the total population for a city and is appropriate for this purpose.",
                "7. Formulate the query to first find the largest city of Canada using 'dbo:largestCity' and then retrieve its population using 'dbo:populationTotal'.",
            ],
            "sparql": "SELECT DISTINCT ?num WHERE { dbr:Canada dbo:largestCity ?city . ?city dbo:populationTotal ?num }",
        },
    },
    {
        "input": """What is the population of Cairo?""",
        "output": {
            "thoughts": [
                "1. The question seeks the population of 'Cairo'.",
                "2. From the context, there are multiple resources labeled 'Cairo', but the correct URI for Cairo, Egypt, is 'dbr:Cairo'.",
                "3. To find the population of Cairo, identify a relevant data property that retrieves population information.",
                "4. In the ontology candidates, 'dbo:populationTotal' is a suitable data property to retrieve the total population of a populated place.",
                "5. Formulate the query to select the population of 'dbr:Cairo' using the 'dbo:populationTotal' property.",
            ],
            "sparql": "SELECT ?pop WHERE { dbr:Cairo dbo:populationTotal ?pop }",
        },
    },
    {
        "input": """How high is the Yokohama Marine Tower?""",
        "output": {
            "thoughts": [
                "1. The question is asking for the height of 'Yokohama Marine Tower'.",
                "2. The context provides a single URI for 'Yokohama Marine Tower', specifically 'dbr:Yokohama_Marine_Tower', which can be used directly in the query.",
                "3. To retrieve the height, identify a data property related to the height or elevation of a structure.",
                "4. In the ontology, 'dbo:height' is a relevant data property that does not have a specific domain or range but matches the information requested.",
                "5. Formulate the query to select the height of 'dbr:Yokohama_Marine_Tower' using the 'dbo:height' property.",
            ],
            "sparql": "SELECT DISTINCT ?num WHERE { dbr:Yokohama_Marine_Tower dbo:height ?num }",
        },
    },
    {
        "input": """Which programming languages were influenced by Perl?""",
        "output": {
            "thoughts": [
                "1. The question seeks to identify programming languages influenced by 'Perl'.",
                "2. Multiple resources were retrieved, but the resource 'dbr:Perl' is the relevant one here as it specifically refers to the programming language.",
                "3. The ontology provides 'dbo:influenced' and 'dbo:influencedBy', both of which relate to influence relationships that could identify languages influenced by Perl.",
                "4. Since the question does not specify a time period or further context, a UNION clause can be used to cover both cases—where Perl influenced other languages and where it is marked as having influenced a language.",
                "5. The query should retrieve distinct URIs of programming languages influenced by Perl by using both 'dbo:influencedBy' and 'dbo:influenced' properties in a UNION pattern.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { { ?uri dbo:influencedBy dbr:Perl } UNION { dbr:Perl dbo:influenced ?uri } }",
        },
    },
    {
        "input": """Give me all Australian nonprofit organizations.""",
        "output": {
            "thoughts": [
                "1. The question asks for nonprofit organizations located in Australia.",
                "2. Multiple URIs for 'Australia' were retrieved, but 'dbr:Australia' best represents the country as a whole.",
                "3. The relevant resources and ontology include 'dbr:Nonprofit_organization' and the class 'dbo:Non-ProfitOrganisation', which indicates nonprofit entities.",
                "4. The properties 'dbo:locationCountry' and 'dbo:location' are useful, with 'dbo:locationCountry' directly linking organizations to their country and 'dbo:location' providing a location reference that can include countries.",
                "5. A UNION pattern can cover both cases to ensure all nonprofits in Australia are captured, either by directly using 'dbo:locationCountry' or by using 'dbo:location' with an additional check for country.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri dbo:type dbr:Nonprofit_organization . { ?uri dbo:locationCountry dbr:Australia } UNION { ?uri dbo:location ?x . ?x dbo:country dbr:Australia } }",
        },
    },
    {
        "input": """Which ingredients do I need for carrot cake?""",
        "output": {
            "thoughts": [
                "1. The question asks for the ingredients needed to make carrot cake.",
                "2. The retrieved resources show 'dbr:Carrot_cake' as the primary URI for 'Carrot cake'.",
                "3. The ontology data includes 'dbo:ingredient' as a relevant object property, which can link food items to their ingredients.",
                "4. Since there is no domain and range for 'dbo:ingredient,' it's likely general enough to use directly in the query.",
                "5. To answer the question, query all ?uri values linked to 'dbr:Carrot_cake' via 'dbo:ingredient'.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { dbr:Carrot_cake dbo:ingredient ?uri }",
        },
    },
    {
        "input": """How tall is Claudia Schiffer?""",
        "output": {
            "thoughts": [
                "1. The question is asking for Claudia Schiffer's height.",
                "2. From the retrieved resources, 'dbr:Claudia_Schiffer' is identified as the URI representing Claudia Schiffer.",
                "3. The ontology data includes 'dbo:height' as a relevant data property, which is commonly used to indicate a person's height.",
                "4. Although there is no domain and range for 'dbo:height,' it should still be appropriate to retrieve the height data directly for the given URI.",
                "5. To answer the question, query the ?height value associated with 'dbr:Claudia_Schiffer' via 'dbo:height'.",
            ],
            "sparql": "SELECT DISTINCT ?height WHERE { dbr:Claudia_Schiffer dbo:height ?height }",
        },
    },
    {
        "input": """In which films did Julia Roberts as well as Richard Gere play?""",
        "output": {
            "thoughts": [
                "1. The question asks for films in which both Julia Roberts and Richard Gere acted together.",
                "2. From the retrieved resources, 'dbr:Julia_Roberts' and 'dbr:Richard_Gere' are the URIs representing Julia Roberts and Richard Gere, respectively.",
                "3. The ontology contains 'dbo:starring' as an object property with the domain 'dbo:Work' and the range 'dbo:Actor,' which is suitable for identifying films with specific actors.",
                "4. Since we're looking for shared appearances, we should query for films (dbo:Film) where both actors are linked through 'dbo:starring'.",
                "5. Formulate the query to select unique film URIs that have both 'dbr:Julia_Roberts' and 'dbr:Richard_Gere' as values for 'dbo:starring'.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri rdf:type dbo:Film ; dbo:starring dbr:Julia_Roberts ; dbo:starring dbr:Richard_Gere }",
        },
    },
    {
        "input": """Who produced the most films?""",
        "output": {
            "thoughts": [
                "1. The question asks for the entity that has produced the most films, suggesting a need to count the number of films linked to each producer.",
                "2. From the resources and ontology, the class 'dbo:Film' represents films, and the property 'dbo:producer' links films to their producers.",
                "3. The query should retrieve entities linked via 'dbo:producer' to instances of 'dbo:Film'.",
                "4. We need to count the films associated with each producer, order by this count in descending order, and select the top result.",
                "5. Formulate the query to select distinct producer URIs, counting the films they produced, and order by the count in descending order to find the producer with the most films.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?film rdf:type dbo:Film . ?film dbo:producer ?uri . } ORDER BY DESC(COUNT(?film)) OFFSET 0 LIMIT 1",
        },
    },
    {
        "input": """Show me all basketball players that are higher than 2 meters.""",
        "output": {
            "thoughts": [
                "1. The question specifies basketball players who are taller than 2 meters, so we need entities of type dbo:BasketballPlayer with a height greater than 2 meters.",
                "2. The ontology provides dbo:BasketballPlayer as the appropriate class for basketball players and dbo:height as the property representing height.",
                "3. The query should use a filter to select only players with heights exceeding 2.0 meters.",
                "4. Structure the query to retrieve unique URIs for basketball players meeting this height criterion.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri a dbo:BasketballPlayer ; dbo:height ?n FILTER ( ?n > 2.0 ) }",
        },
    },
    {
        "input": """How many films did Leonardo DiCaprio star in?""",
        "output": {
            "thoughts": [
                "1. The question requires counting the number of films Leonardo DiCaprio starred in.",
                "2. We need to retrieve entities of type dbo:Film, where Leonardo DiCaprio is listed as the star.",
                "3. The dbo:starring property links films to actors, so we can use this property with dbr:Leonardo_DiCaprio as the object.",
                "4. To get the count, we use COUNT(DISTINCT ?uri) to ensure each film is counted only once, and return the result as ?c.",
            ],
            "sparql": "SELECT (COUNT(DISTINCT ?uri) AS ?c) WHERE { ?uri a dbo:Film ; dbo:starring dbr:Leonardo_DiCaprio }",
        },
    },
    {
        "input": """Give me all companies in the advertising industry.""",
        "output": {
            "thoughts": [
                "1. The question asks for companies in the advertising industry.",
                "2. We can identify companies using the dbo:Company class.",
                "3. The dbo:industry property links a company to its industry, and dbr:Advertising can represent the advertising industry.",
                "4. To ensure we capture relevant companies even if their industry isn't explicitly 'Advertising,' we add a regex filter to match industries with 'advertising' in the label.",
                "5. The query structure includes both direct matches with dbr:Advertising and regex matches to ensure broader coverage.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri a dbo:Company { ?uri dbo:industry dbr:Advertising } UNION { ?uri dbo:industry ?industry FILTER regex(?industry, 'advertising', 'i') } }",
        },
    },
    {
        "input": """How many movies did Park Chan-wook direct?""",
        "output": {
            "thoughts": [
                "1. The question asks for the number of movies directed by Park Chan-wook.",
                "2. Park Chan-wook is represented by the URI dbr:Park_Chan-wook.",
                "3. The dbo:director property associates a film with its director.",
                "4. We need to count the unique films where Park Chan-wook is the director, using the dbo:Film class as the target entity.",
            ],
            "sparql": "SELECT COUNT(DISTINCT ?uri AS ?uri) WHERE { ?uri dbo:director dbr:Park_Chan-wook . }",
        },
    },
    {
        "input": """Who wrote the Game of Thrones theme?""",
        "output": {
            "thoughts": [
                "1. The question asks for the composer of the Game of Thrones theme.",
                "2. The resource for Game of Thrones is identified by the URI 'dbr:Game_of_Thrones'.",
                "3. The relevant property to identify the composer of a theme or soundtrack is 'dbo:composer', which links a work with its creator or composer.",
                "4. The query should retrieve the URI of the composer associated with the Game of Thrones series.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { dbr:Game_of_Thrones dbo:composer ?uri }",
        },
    },
]


WIKIDATA_GENERATE_SPARQL_FEW_SHOTS = [
    {
        "input": """How many inhabitants does the largest city in Canada have?""",
        "output": {
            "thoughts": [
                "1. The question asks for the population of the largest city in Canada.",
                "2. To retrieve this information from Wikidata, identify the correct URI for 'Canada' and the relevant properties associated with cities and population.",
                "3. From the provided URIs, 'wd:Q16' is the correct identifier for the country 'Canada' in Wikidata.",
                "4. The term 'largest city' requires identifying Canadian cities and sorting them by population to determine the largest.",
                "5. Based on the ontology, the 'wdt:P17' property links entities to their respective countries, which helps identify cities in Canada.",
                "6. The 'wdt:P31/wdt:P279* wd:Q515' construct allows filtering for entities that are instances of or subclasses of 'city' (Q515).",
                "7. The 'wdt:P1082' property specifies the population of cities, which is necessary to sort and retrieve the largest city by population.",
                "8. Formulate a SPARQL query to find all cities in Canada, retrieve their populations using 'wdt:P1082', and sort the results in descending order to determine the largest city by population.",
            ],
            "sparql": "SELECT DISTINCT ?val WHERE { ?uri wdt:P17 wd:Q16 ; wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?val } ORDER BY DESC(?val) LIMIT 1",
        },
    },
    {
        "input": """What is the population of Cairo?""",
        "output": {
            "thoughts": [
                "1. The question seeks the population of 'Cairo'.",
                "2. From the retrieved resources, the correct URI for Cairo, the capital city of Egypt, is 'wd:Q85' in Wikidata.",
                "3. To retrieve the population, we need to identify a relevant property that provides population data.",
                "4. From the retrieved ontology, the property 'wdt:P1082' represents the total population of an entity and is suitable for this query.",
                "5. Formulate a SPARQL query to retrieve the population of 'wd:Q85' using the 'wdt:P1082' property.",
            ],
            "sparql": "SELECT DISTINCT ?val WHERE { wd:Q85 wdt:P1082 ?val . }",
        },
    },
    {
        "input": """How high is the Yokohama Marine Tower?""",
        "output": {
            "thoughts": [
                "1. The question seeks the height of the 'Yokohama Marine Tower'.",
                "2. From the retrieved resources, the correct URI for 'Yokohama Marine Tower' is 'wd:Q1207989' in Wikidata.",
                "3. To retrieve the height, a property representing height information is required.",
                "4. In the retrieved ontology, the property 'wdt:P2048' represents the height of a structure and is appropriate for this query.",
                "5. Formulate a SPARQL query to retrieve the height of 'wd:Q1207989' using the 'wdt:P2048' property.",
            ],
            "sparql": "SELECT DISTINCT ?num WHERE { wd:Q1207989 wdt:P2048 ?num . }",
        },
    },
    {
        "input": """Which programming languages were influenced by Perl?""",
        "output": {
            "thoughts": [
                "1. The question seeks to identify programming languages influenced by 'Perl'.",
                "2. From the retrieved resources, the correct URI for 'Perl' as a programming language is 'wd:Q42478'.",
                "3. To find programming languages influenced by Perl, a property representing influence relationships is needed.",
                "4. In the retrieved ontology, the property 'wdt:P737' represents the influence relationship, where the subject is influenced by the object.",
                "5. A SPARQL query can be formulated to retrieve entities classified as programming languages (wd:Q9143) that are influenced by 'wd:Q42478' using the 'wdt:P737' property.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q9143 ; wdt:P737 wd:Q42478 . }",
        },
    },
    {
        "input": """Give me all Australian nonprofit organizations.""",
        "output": {
            "thoughts": [
                "1. The question seeks all nonprofit organizations located in Australia.",
                "2. From the retrieved resources, the correct URI for 'Australia' as a country is 'wd:Q408'.",
                "3. For 'nonprofit organization', the correct URI is 'wd:Q163740', representing organizations operated for collective benefit.",
                "4. To find Australian nonprofit organizations, entities classified as 'nonprofit organizations' (wd:Q163740) should be retrieved and filtered based on their location being in Australia (wd:Q408).",
                "5. The properties 'wdt:P31' (instance of), 'wdt:P17' (country), and 'wdt:P159' (headquarters location) are relevant for identifying nonprofit organizations in Australia.",
                "6. A SPARQL query can be formulated to retrieve all entities that are instances of 'nonprofit organization' (wd:Q163740) and are either located in Australia (wdt:P17 wd:Q408) or have their headquarters there (wdt:P159 wd:Q408).",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q163740 . { ?uri wdt:P159|wdt:P17 wd:Q408 . } UNION { ?uri wdt:P159|wdt:P17 ?city . ?city wdt:P159 wd:Q408 } }",
        },
    },
    {
        "input": """Which ingredients do I need for carrot cake?""",
        "output": {
            "thoughts": [
                "1. The question seeks the ingredients needed to make carrot cake.",
                "2. From the retrieved resources, the correct URI for 'Carrot cake' as a sweet dish is 'wd:Q656071'.",
                "3. To identify the ingredients, a property that specifies components or ingredients is required.",
                "4. In the retrieved ontology, the property 'wdt:P527' represents components or parts of an entity, making it suitable for retrieving ingredients.",
                "5. A SPARQL query can be formulated to retrieve all entities linked to 'wd:Q656071' through the 'wdt:P527' property, representing the ingredients of carrot cake.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { wd:Q656071 wdt:P527 ?uri . }",
        },
    },
    {
        "input": """How tall is Claudia Schiffer?""",
        "output": {
            "thoughts": [
                "1. The question seeks the height of Claudia Schiffer.",
                "2. The retrieved resources do not include Claudia Schiffer explicitly, but her Wikidata URI can be identified as 'wd:Q76717' based on external context.",
                "3. To find her height, a property representing physical height is required.",
                "4. In the retrieved ontology, the property 'wdt:P2048' represents height and is suitable for retrieving this information.",
                "5. A SPARQL query can be formulated to retrieve the height value for 'wd:Q76717' using the 'wdt:P2048' property.",
            ],
            "sparql": "SELECT DISTINCT ?val WHERE { wd:Q76717 wdt:P2048 ?val . }",
        },
    },
    {
        "input": """In which films did Julia Roberts as well as Richard Gere play?""",
        "output": {
            "thoughts": [
                "1. The question seeks films in which both Julia Roberts and Richard Gere acted.",
                "2. From the retrieved resources, 'wd:Q40523' is the correct URI for Julia Roberts, and 'wd:Q48410' is the correct URI for Richard Gere.",
                "3. To identify shared films, entities classified as 'films' (wd:Q11424) must be retrieved and filtered based on the two actors being part of the cast.",
                "4. The property 'wdt:P161' (cast member) is relevant for linking actors to films.",
                "5. A SPARQL query can be formulated to retrieve all film entities where both 'wd:Q40523' (Julia Roberts) and 'wd:Q48410' (Richard Gere) appear as cast members using the 'wdt:P161' property.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q11424 . ?uri wdt:P161 wd:Q48410 . ?uri wdt:P161 wd:Q40523 . }",
        },
    },
    {
        "input": """Who produced the most films?""",
        "output": {
            "thoughts": [
                "1. The question asks for films in which both Julia Roberts and Richard Gere acted together.",
                "2. From the retrieved resources, 'wd:Q40523' and 'wd:Q48410' are the Wikidata URIs representing Julia Roberts and Richard Gere, respectively.",
                "3. The ontology contains 'P161' as a property indicating 'cast member,' which is suitable for identifying films with specific actors.",
                "4. Since we're looking for shared appearances, we need to query for films (instances of Q11424, which represents 'film') where both actors are listed as values for 'P161' (cast member).",
                "5. Formulate the query to select distinct film URIs that have both 'wd:Q40523' (Julia Roberts) and 'wd:Q48410' (Richard Gere) as values for the 'P161' property.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?film wdt:P31 wd:Q11424 . ?film wdt:P162 ?uri . } GROUP BY ?uri ORDER BY DESC(COUNT(?film)) LIMIT 1",
        },
    },
    {
        "input": """Show me all basketball players that are higher than 2 meters.""",
        "output": {
            "thoughts": [
                "1. The question asks for basketball players taller than 2 meters, so we need to identify individuals of the relevant type and with height data available.",
                "2. From the retrieved resources, basketball players can be represented by the entity class 'wd:Q3665646' (basketball player).",
                "3. The ontology provides 'P2048' as the property for height, which can be used to determine the physical height of individuals.",
                "4. A filter must be applied to retrieve only players whose height exceeds 200 centimeters (2 meters).",
                "5. Formulate the query to select distinct basketball player URIs where the 'P2048' value is greater than 200.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri wdt:P106 wd:Q3665646 . ?uri wdt:P2048 ?height .  FILTER(?height > 200)}",
        },
    },
    {
        "input": """How many films did Leonardo DiCaprio star in?""",
        "output": {
            "thoughts": [
                "1. The question asks for the total number of films Leonardo DiCaprio has starred in.",
                "2. Leonardo DiCaprio is represented by the Wikidata URI 'wd:Q38111'.",
                "3. Films are identified as instances of 'wd:Q11424' (film), and 'P161' (cast member) links films to their actors.",
                "4. To count the films Leonardo DiCaprio has starred in, we must query for distinct film URIs where he is listed as a value for the 'P161' property.",
                "5. Use the COUNT(DISTINCT ?uri) function in the query to count the number of unique films and return the result as ?c.",
            ],
            "sparql": "SELECT (COUNT(DISTINCT ?uri) AS ?c) WHERE { ?uri wdt:P31 wd:Q11424 ; wdt:P161 wd:Q38111 . }",
        },
    },
    {
        "input": """Give me all companies in the advertising industry.""",
        "output": {
            "thoughts": [
                "1. The question seeks companies that operate in the advertising industry.",
                "2. Companies can be represented by the Wikidata class 'wd:Q783794' (company).",
                "3. The property 'P452' (industry) links companies to the industries they are involved in.",
                "4. The advertising industry can be represented by the entity 'wd:Q37038' (advertising).",
                "5. Formulate the query to retrieve unique URIs for companies ('wd:Q783794') where the 'P452' property value matches 'wd:Q37038' (advertising).",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q4830453 ; wdt:P452 wd:Q39809 . }",
        },
    },
    {
        "input": """How many movies did Park Chan-wook direct?""",
        "output": {
            "thoughts": [
                "1. The question asks for the total number of movies directed by Park Chan-wook.",
                "2. Park Chan-wook is represented by the Wikidata URI 'wd:Q315484'.",
                "3. The property 'P57' (director) links films to their directors, making it appropriate for this query.",
                "4. The goal is to count distinct film entities ('wd:Q11424', representing films) where Park Chan-wook is listed as the director using 'P57'.",
                "5. Use the COUNT(DISTINCT ?uri) function to determine the number of unique films directed by Park Chan-wook and return the result as ?count.",
            ],
            "sparql": "SELECT (COUNT(DISTINCT ?uri) as ?count) WHERE {  ?uri wdt:P57 wd:Q315484 . }",
        },
    },
    {
        "input": """Who wrote the Game of Thrones theme?""",
        "output": {
            "thoughts": [
                "1. The question asks for the composer of the Game of Thrones theme.",
                "2. The specific entity representing the 'Game of Thrones Theme' is identified by the URI 'wd:Q27813929'.",
                "3. The relevant property to find the composer is 'P86' (composer), which links musical works to their composers.",
                "4. The query should retrieve the URI of the composer associated with the 'Game of Thrones Theme' using the property 'P86'.",
            ],
            "sparql": "SELECT DISTINCT ?uri WHERE { wd:Q23572 wdt:P86 ?uri . }",
        },
    },
]

# DBPEDIA_GENERATE_SPARQL_FEW_SHOTS = [
#     {
#         "input": """How many inhabitants does the largest city in Canada have?""",
#         "output": """Thoughts:
# 1. The question asks for the population of the largest city in Canada.
# 2. To retrieve this information from DBpedia, identify the correct URI for 'Canada' and the relevant properties associated with cities and population.
# 3. From the context, 'Canada' has multiple possible URIs, with 'dbr:Canada' being the correct general URI.
# 4. The term 'largest city' suggests we should find the entity linked to Canada that represents its largest city, which requires identifying the property that connects 'Canada' to its largest city.
# 5. From the list of ontology candidates, 'dbo:largestCity' is the most relevant object property for retrieving the largest city of a place, with both the domain and range indicating populated places.
# 6. After obtaining the largest city, we need to retrieve its population. The 'dbo:populationTotal' data property provides the total population for a city and is appropriate for this purpose.
# 7. Formulate the query to first find the largest city of Canada using 'dbo:largestCity' and then retrieve its population using 'dbo:populationTotal'.
# SPARQL:
# ```SELECT DISTINCT ?num WHERE { dbr:Canada dbo:largestCity ?city . ?city dbo:populationTotal ?num }```""",
#     },
#     {
#         "input": """What is the population of Cairo?""",
#         "output": """Thoughts:
# 1. The question seeks the population of 'Cairo'.
# 2. From the context, there are multiple resources labeled 'Cairo', but the correct URI for Cairo, Egypt, is 'dbr:Cairo'.
# 3. To find the population of Cairo, identify a relevant data property that retrieves population information.
# 4. In the ontology candidates, 'dbo:populationTotal' is a suitable data property to retrieve the total population of a populated place.
# 5. Formulate the query to select the population of 'dbr:Cairo' using the 'dbo:populationTotal' property.
# SPARQL:
# ```SELECT ?pop WHERE { dbr:Cairo dbo:populationTotal ?pop }```""",
#     },
#     {
#         "input": """How high is the Yokohama Marine Tower?""",
#         "output": """Thoughts:
# 1. The question is asking for the height of 'Yokohama Marine Tower'.
# 2. The context provides a single URI for 'Yokohama Marine Tower', specifically 'dbr:Yokohama_Marine_Tower', which can be used directly in the query.
# 3. To retrieve the height, identify a data property related to the height or elevation of a structure.
# 4. In the ontology, 'dbo:height' is a relevant data property that does not have a specific domain or range but matches the information requested.
# 5. Formulate the query to select the height of 'dbr:Yokohama_Marine_Tower' using the 'dbo:height' property.
# SPARQL:
# ```SELECT DISTINCT ?num WHERE { dbr:Yokohama_Marine_Tower dbo:height ?num }```""",
#     },
#     {
#         "input": """Which programming languages were influenced by Perl?""",
#         "output": """Thoughts:
# 1. The question seeks to identify programming languages influenced by 'Perl'.
# 2. Multiple resources were retrieved, but the resource 'dbr:Perl' is the relevant one here as it specifically refers to the programming language.
# 3. The ontology provides 'dbo:influenced' and 'dbo:influencedBy', both of which relate to influence relationships that could identify languages influenced by Perl.
# 4. Since the question does not specify a time period or further context, a UNION clause can be used to cover both cases—where Perl influenced other languages and where it is marked as having influenced a language.
# 5. The query should retrieve distinct URIs of programming languages influenced by Perl by using both 'dbo:influencedBy' and 'dbo:influenced' properties in a UNION pattern.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { { ?uri dbo:influencedBy dbr:Perl } UNION { dbr:Perl dbo:influenced ?uri } }```""",
#     },
#     {
#         "input": """Give me all Australian nonprofit organizations.""",
#         "output": """Thoughts:
# 1. The question asks for nonprofit organizations located in Australia.
# 2. Multiple URIs for 'Australia' were retrieved, but 'dbr:Australia' best represents the country as a whole.
# 3. The relevant resources and ontology include 'dbr:Nonprofit_organization' and the class 'dbo:Non-ProfitOrganisation', which indicates nonprofit entities.
# 4. The properties 'dbo:locationCountry' and 'dbo:location' are useful, with 'dbo:locationCountry' directly linking organizations to their country and 'dbo:location' providing a location reference that can include countries.
# 5. A UNION pattern can cover both cases to ensure all nonprofits in Australia are captured, either by directly using 'dbo:locationCountry' or by using 'dbo:location' with an additional check for country.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri dbo:type dbr:Nonprofit_organization . { ?uri dbo:locationCountry dbr:Australia } UNION { ?uri dbo:location ?x . ?x dbo:country dbr:Australia } }```""",
#     },
#     {
#         "input": """Which ingredients do I need for carrot cake?""",
#         "output": """Thoughts:
# 1. The question asks for the ingredients needed to make carrot cake.
# 2. The retrieved resources show 'dbr:Carrot_cake' as the primary URI for 'Carrot cake'.
# 3. The ontology data includes 'dbo:ingredient' as a relevant object property, which can link food items to their ingredients.
# 4. Since there is no domain and range for 'dbo:ingredient,' it's likely general enough to use directly in the query.
# 5. To answer the question, query all ?uri values linked to 'dbr:Carrot_cake' via 'dbo:ingredient'.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { dbr:Carrot_cake dbo:ingredient ?uri }```""",
#     },
#     {
#         "input": """How tall is Claudia Schiffer?""",
#         "output": """Thoughts:
# 1. The question is asking for Claudia Schiffer's height.
# 2. From the retrieved resources, 'dbr:Claudia_Schiffer' is identified as the URI representing Claudia Schiffer.
# 3. The ontology data includes 'dbo:height' as a relevant data property, which is commonly used to indicate a person's height.
# 4. Although there is no domain and range for 'dbo:height,' it should still be appropriate to retrieve the height data directly for the given URI.
# 5. To answer the question, query the ?height value associated with 'dbr:Claudia_Schiffer' via 'dbo:height'.
# SPARQL:
# ```SELECT DISTINCT ?height WHERE { dbr:Claudia_Schiffer dbo:height ?height }```""",
#     },
#     {
#         "input": """In which films did Julia Roberts as well as Richard Gere play?""",
#         "output": """Thoughts:
# 1. The question asks for films in which both Julia Roberts and Richard Gere acted together.
# 2. From the retrieved resources, 'dbr:Julia_Roberts' and 'dbr:Richard_Gere' are the URIs representing Julia Roberts and Richard Gere, respectively.
# 3. The ontology contains 'dbo:starring' as an object property with the domain 'dbo:Work' and the range 'dbo:Actor,' which is suitable for identifying films with specific actors.
# 4. Since we're looking for shared appearances, we should query for films (dbo:Film) where both actors are linked through 'dbo:starring'.
# 5. Formulate the query to select unique film URIs that have both 'dbr:Julia_Roberts' and 'dbr:Richard_Gere' as values for 'dbo:starring'.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri rdf:type dbo:Film ; dbo:starring dbr:Julia_Roberts ; dbo:starring dbr:Richard_Gere }```""",
#     },
#     {
#         "input": """Who produced the most films?""",
#         "output": """Thoughts:
# 1. The question asks for the entity that has produced the most films, suggesting a need to count the number of films linked to each producer.
# 2. From the resources and ontology, the class 'dbo:Film' represents films, and the property 'dbo:producer' links films to their producers.
# 3. The query should retrieve entities linked via 'dbo:producer' to instances of 'dbo:Film'.
# 4. We need to count the films associated with each producer, order by this count in descending order, and select the top result.
# 5. Formulate the query to select distinct producer URIs, counting the films they produced, and order by the count in descending order to find the producer with the most films.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?film rdf:type dbo:Film . ?film dbo:producer ?uri . } ORDER BY DESC(COUNT(?film)) OFFSET 0 LIMIT 1```""",
#     },
#     {
#         "input": """Show me all basketball players that are higher than 2 meters.""",
#         "output": """Thoughts:
# 1. The question specifies basketball players who are taller than 2 meters, so we need entities of type dbo:BasketballPlayer with a height greater than 2 meters.
# 2. The ontology provides dbo:BasketballPlayer as the appropriate class for basketball players and dbo:height as the property representing height.
# 3. The query should use a filter to select only players with heights exceeding 2.0 meters.
# 4. Structure the query to retrieve unique URIs for basketball players meeting this height criterion.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri a dbo:BasketballPlayer ; dbo:height ?n FILTER ( ?n > 2.0 ) }```""",
#     },
#     {
#         "input": """How many films did Leonardo DiCaprio star in?""",
#         "output": """Thoughts:
# 1. The question requires counting the number of films Leonardo DiCaprio starred in.
# 2. We need to retrieve entities of type dbo:Film, where Leonardo DiCaprio is listed as the star.
# 3. The dbo:starring property links films to actors, so we can use this property with dbr:Leonardo_DiCaprio as the object.
# 4. To get the count, we use COUNT(DISTINCT ?uri) to ensure each film is counted only once, and return the result as ?c.
# SPARQL:
# ```SELECT (COUNT(DISTINCT ?uri) AS ?c) WHERE { ?uri a dbo:Film ; dbo:starring dbr:Leonardo_DiCaprio }```""",
#     },
#     {
#         "input": """Give me all companies in the advertising industry.""",
#         "output": """Thoughts:
# 1. The question asks for companies in the advertising industry.
# 2. We can identify companies using the dbo:Company class.
# 3. The dbo:industry property links a company to its industry, and dbr:Advertising can represent the advertising industry.
# 4. To ensure we capture relevant companies even if their industry isn't explicitly 'Advertising,' we add a regex filter to match industries with 'advertising' in the label.
# 5. The query structure includes both direct matches with dbr:Advertising and regex matches to ensure broader coverage.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri a dbo:Company { ?uri dbo:industry dbr:Advertising } UNION { ?uri dbo:industry ?industry FILTER regex(?industry, 'advertising', 'i') } }```""",
#     },
#     {
#         "input": """How many movies did Park Chan-wook direct?""",
#         "output": """Thoughts:
# 1. The question asks for the number of movies directed by Park Chan-wook.
# 2. Park Chan-wook is represented by the URI dbr:Park_Chan-wook.
# 3. The dbo:director property associates a film with its director.
# 4. We need to count the unique films where Park Chan-wook is the director, using the dbo:Film class as the target entity.
# SPARQL:
# ```SELECT COUNT(DISTINCT ?uri AS ?uri) WHERE { ?uri dbo:director dbr:Park_Chan-wook . }```""",
#     },
#     {
#         "input": """Who wrote the Game of Thrones theme?""",
#         "output": """Thoughts:
# 1. The question asks for the composer of the Game of Thrones theme.
# 2. The resource for Game of Thrones is identified by the URI 'dbr:Game_of_Thrones'.
# 3. The relevant property to identify the composer of a theme or soundtrack is 'dbo:composer', which links a work with its creator or composer.
# 4. The query should retrieve the URI of the composer associated with the Game of Thrones series.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { dbr:Game_of_Thrones dbo:composer ?uri }```""",
#     },
# ]

# WIKIDATA_GENERATE_SPARQL_FEW_SHOTS = [
#     {
#         "input": """How many inhabitants does the largest city in Canada have?""",
#         "output": """Thoughts:
# 1. The question asks for the population of the largest city in Canada.
# 2. To retrieve this information from Wikidata, identify the correct URI for 'Canada' and the relevant properties associated with cities and population.
# 3. From the provided URIs, 'wd:Q16' is the correct identifier for the country 'Canada' in Wikidata.
# 4. The term 'largest city' requires identifying Canadian cities and sorting them by population to determine the largest.
# 5. Based on the ontology, the 'wdt:P17' property links entities to their respective countries, which helps identify cities in Canada.
# 6. The 'wdt:P31/wdt:P279* wd:Q515' construct allows filtering for entities that are instances of or subclasses of 'city' (Q515).
# 7. The 'wdt:P1082' property specifies the population of cities, which is necessary to sort and retrieve the largest city by population.
# 8. Formulate a SPARQL query to find all cities in Canada, retrieve their populations using 'wdt:P1082', and sort the results in descending order to determine the largest city by population.
# SPARQL:
# ```SELECT DISTINCT ?val WHERE { ?uri wdt:P17 wd:Q16 ; wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?val } ORDER BY DESC(?val) LIMIT 1```""",
#     },
#     {
#         "input": """What is the population of Cairo?""",
#         "output": """Thoughts:
# 1. The question seeks the population of 'Cairo'.
# 2. From the retrieved resources, the correct URI for Cairo, the capital city of Egypt, is 'wd:Q85' in Wikidata.
# 3. To retrieve the population, we need to identify a relevant property that provides population data.
# 4. From the retrieved ontology, the property 'wdt:P1082' represents the total population of an entity and is suitable for this query.
# 5. Formulate a SPARQL query to retrieve the population of 'wd:Q85' using the 'wdt:P1082' property.
# SPARQL:
# ```SELECT DISTINCT ?val WHERE { wd:Q85 wdt:P1082 ?val . }```""",
#     },
#     {
#         "input": """How high is the Yokohama Marine Tower?""",
#         "output": """Thoughts:
# 1. The question seeks the height of the 'Yokohama Marine Tower'.
# 2. From the retrieved resources, the correct URI for 'Yokohama Marine Tower' is 'wd:Q1207989' in Wikidata.
# 3. To retrieve the height, a property representing height information is required.
# 4. In the retrieved ontology, the property 'wdt:P2048' represents the height of a structure and is appropriate for this query.
# 5. Formulate a SPARQL query to retrieve the height of 'wd:Q1207989' using the 'wdt:P2048' property.
# SPARQL:
# ```SELECT DISTINCT ?num WHERE { wd:Q1207989 wdt:P2048 ?num . }```""",
#     },
#     {
#         "input": """Which programming languages were influenced by Perl?""",
#         "output": """Thoughts:
# 1. The question seeks to identify programming languages influenced by 'Perl'.
# 2. From the retrieved resources, the correct URI for 'Perl' as a programming language is 'wd:Q42478'.
# 3. To find programming languages influenced by Perl, a property representing influence relationships is needed.
# 4. In the retrieved ontology, the property 'wdt:P737' represents the influence relationship, where the subject is influenced by the object.
# 5. A SPARQL query can be formulated to retrieve entities classified as programming languages (wd:Q9143) that are influenced by 'wd:Q42478' using the 'wdt:P737' property.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q9143 ; wdt:P737 wd:Q42478 . }```""",
#     },
#     {
#         "input": """Give me all Australian nonprofit organizations.""",
#         "output": """Thoughts:
# 1. The question seeks all nonprofit organizations located in Australia.
# 2. From the retrieved resources, the correct URI for 'Australia' as a country is 'wd:Q408'.
# 3. For 'nonprofit organization', the correct URI is 'wd:Q163740', representing organizations operated for collective benefit.
# 4. To find Australian nonprofit organizations, entities classified as 'nonprofit organizations' (wd:Q163740) should be retrieved and filtered based on their location being in Australia (wd:Q408).
# 5. The properties 'wdt:P31' (instance of), 'wdt:P17' (country), and 'wdt:P159' (headquarters location) are relevant for identifying nonprofit organizations in Australia.
# 6. A SPARQL query can be formulated to retrieve all entities that are instances of 'nonprofit organization' (wd:Q163740) and are either located in Australia (wdt:P17 wd:Q408) or have their headquarters there (wdt:P159 wd:Q408).
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q163740 . { ?uri wdt:P159|wdt:P17 wd:Q408 . } UNION { ?uri wdt:P159|wdt:P17 ?city . ?city wdt:P159 wd:Q408 } }```""",
#     },
#     {
#         "input": """Which ingredients do I need for carrot cake?""",
#         "output": """Thoughts:
# 1. The question seeks the ingredients needed to make carrot cake.
# 2. From the retrieved resources, the correct URI for 'Carrot cake' as a sweet dish is 'wd:Q656071'.
# 3. To identify the ingredients, a property that specifies components or ingredients is required.
# 4. In the retrieved ontology, the property 'wdt:P527' represents components or parts of an entity, making it suitable for retrieving ingredients.
# 5. A SPARQL query can be formulated to retrieve all entities linked to 'wd:Q656071' through the 'wdt:P527' property, representing the ingredients of carrot cake.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { wd:Q656071 wdt:P527 ?uri . }```""",
#     },
#     {
#         "input": """How tall is Claudia Schiffer?""",
#         "output": """Thoughts:
# 1. The question seeks the height of Claudia Schiffer.
# 2. The retrieved resources do not include Claudia Schiffer explicitly, but her Wikidata URI can be identified as 'wd:Q76717' based on external context.
# 3. To find her height, a property representing physical height is required.
# 4. In the retrieved ontology, the property 'wdt:P2048' represents height and is suitable for retrieving this information.
# 5. A SPARQL query can be formulated to retrieve the height value for 'wd:Q76717' using the 'wdt:P2048' property.
# SPARQL:
# ```SELECT DISTINCT ?val WHERE { wd:Q76717 wdt:P2048 ?val . }```""",
#     },
#     {
#         "input": """In which films did Julia Roberts as well as Richard Gere play?""",
#         "output": """Thoughts:
# 1. The question seeks films in which both Julia Roberts and Richard Gere acted.
# 2. From the retrieved resources, 'wd:Q40523' is the correct URI for Julia Roberts, and 'wd:Q48410' is the correct URI for Richard Gere.
# 3. To identify shared films, entities classified as 'films' (wd:Q11424) must be retrieved and filtered based on the two actors being part of the cast.
# 4. The property 'wdt:P161' (cast member) is relevant for linking actors to films.
# 5. A SPARQL query can be formulated to retrieve all film entities where both 'wd:Q40523' (Julia Roberts) and 'wd:Q48410' (Richard Gere) appear as cast members using the 'wdt:P161' property.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q11424 . ?uri wdt:P161 wd:Q48410 . ?uri wdt:P161 wd:Q40523 . }```""",
#     },
#     {
#         "input": """Who produced the most films?""",
#         "output": """Thoughts:
# 1. The question asks for films in which both Julia Roberts and Richard Gere acted together.
# 2. From the retrieved resources, 'wd:Q40523' and 'wd:Q48410' are the Wikidata URIs representing Julia Roberts and Richard Gere, respectively.
# 3. The ontology contains 'P161' as a property indicating 'cast member,' which is suitable for identifying films with specific actors.
# 4. Since we're looking for shared appearances, we need to query for films (instances of Q11424, which represents 'film') where both actors are listed as values for 'P161' (cast member).
# 5. Formulate the query to select distinct film URIs that have both 'wd:Q40523' (Julia Roberts) and 'wd:Q48410' (Richard Gere) as values for the 'P161' property.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?film wdt:P31 wd:Q11424 . ?film wdt:P162 ?uri . } GROUP BY ?uri ORDER BY DESC(COUNT(?film)) LIMIT 1```""",
#     },
#     {
#         "input": """Show me all basketball players that are higher than 2 meters.""",
#         "output": """Thoughts:
# 1. The question asks for basketball players taller than 2 meters, so we need to identify individuals of the relevant type and with height data available.
# 2. From the retrieved resources, basketball players can be represented by the entity class 'wd:Q3665646' (basketball player).
# 3. The ontology provides 'P2048' as the property for height, which can be used to determine the physical height of individuals.
# 4. A filter must be applied to retrieve only players whose height exceeds 200 centimeters (2 meters).
# 5. Formulate the query to select distinct basketball player URIs where the 'P2048' value is greater than 200.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri wdt:P106 wd:Q3665646 . ?uri wdt:P2048 ?height .  FILTER(?height > 200)}```""",
#     },
#     {
#         "input": """How many films did Leonardo DiCaprio star in?""",
#         "output": """Thoughts:
# 1. The question asks for the total number of films Leonardo DiCaprio has starred in.
# 2. Leonardo DiCaprio is represented by the Wikidata URI 'wd:Q38111'.
# 3. Films are identified as instances of 'wd:Q11424' (film), and 'P161' (cast member) links films to their actors.
# 4. To count the films Leonardo DiCaprio has starred in, we must query for distinct film URIs where he is listed as a value for the 'P161' property.
# 5. Use the COUNT(DISTINCT ?uri) function in the query to count the number of unique films and return the result as ?c.
# SPARQL:
# ```SELECT (COUNT(DISTINCT ?uri) AS ?c) WHERE { ?uri wdt:P31 wd:Q11424 ; wdt:P161 wd:Q38111 . }```""",
#     },
#     {
#         "input": """Give me all companies in the advertising industry.""",
#         "output": """Thoughts:
# 1. The question seeks companies that operate in the advertising industry.
# 2. Companies can be represented by the Wikidata class 'wd:Q783794' (company).
# 3. The property 'P452' (industry) links companies to the industries they are involved in.
# 4. The advertising industry can be represented by the entity 'wd:Q37038' (advertising).
# 5. Formulate the query to retrieve unique URIs for companies ('wd:Q783794') where the 'P452' property value matches 'wd:Q37038' (advertising).
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q4830453 ; wdt:P452 wd:Q39809 . }```""",
#     },
#     {
#         "input": """How many movies did Park Chan-wook direct?""",
#         "output": """Thoughts:
# 1. The question asks for the total number of movies directed by Park Chan-wook.
# 2. Park Chan-wook is represented by the Wikidata URI 'wd:Q315484'.
# 3. The property 'P57' (director) links films to their directors, making it appropriate for this query.
# 4. The goal is to count distinct film entities ('wd:Q11424', representing films) where Park Chan-wook is listed as the director using 'P57'.
# 5. Use the COUNT(DISTINCT ?uri) function to determine the number of unique films directed by Park Chan-wook and return the result as ?count.
# SPARQL:
# ```SELECT (COUNT(DISTINCT ?uri) as ?count) WHERE {  ?uri wdt:P57 wd:Q315484 . }```""",
#     },
#     {
#         "input": """Who wrote the Game of Thrones theme?""",
#         "output": """Thoughts:
# 1. The question asks for the composer of the Game of Thrones theme.
# 2. The specific entity representing the 'Game of Thrones Theme' is identified by the URI 'wd:Q27813929'.
# 3. The relevant property to find the composer is 'P86' (composer), which links musical works to their composers.
# 4. The query should retrieve the URI of the composer associated with the 'Game of Thrones Theme' using the property 'P86'.
# SPARQL:
# ```SELECT DISTINCT ?uri WHERE { wd:Q23572 wdt:P86 ?uri . }```""",
#     },
# ]


for fs in [
    EXTRACT_ENTITY_FEW_SHOTS,
    INTENT_CLASSIFICATION_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
    DBPEDIA_GENERATE_SPARQL_FEW_SHOTS,
    WIKIDATA_GENERATE_SPARQL_FEW_SHOTS,
]:
    for f in fs:
        f["output"] = json.dumps(f["output"], indent=4)
