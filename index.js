import dotenv from 'dotenv';
dotenv.config();
import express from 'express';
import {AzureChatOpenAI, AzureOpenAIEmbeddings} from "@langchain/openai";
import {HumanMessage, SystemMessage, AIMessage} from "@langchain/core/messages"
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FaissStore } from "@langchain/community/vectorstores/faiss";


let vectorStore
const model = new AzureChatOpenAI({ temperature: 1 });
const embeddings = new AzureOpenAIEmbeddings({
    temperature: 0,
    azureOpenAIApiEmbeddingsDeploymentName: process.env.AZURE_EMBEDDING_DEPLOYMENT_NAME
});


// const loader = new PDFLoader('./documents/reactiesommen.pdf');
// const docs = await loader.load();
// const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 50 });
// const splitDocs = await textSplitter.splitDocuments(docs);
// console.log(`Document split into ${splitDocs.length} chunks. Now saving into vector store`);
// vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
// await vectorStore.save("./vectordatabase"); // geef hier de naam van de directory waar je de data gaat opslaan


const app = express();
const port = 8000;
import cors from 'cors'

app.use(express.urlencoded({ extended: true })); // voor x-www-form-urlencoded
app.use(express.json());
app.use(cors()); // ✅ Sta alles toe (voor dev prima)

vectorStore = await FaissStore.load("./vectordatabase", embeddings);

app.post('/question', async (req, res) => {

    let system = "Je bent een onderwijsdeskundige die verhaaltjesvragen maakt voor kinderen van de basisschool uit verschillende groepen." +
        "Houd rekening met de verschillende groepen, als een kind in een hogere groep zit moet de vraag moeilijker zijn, maar als een kind in een lagere groep zit moet de vraag makkelijker zijn" +
        "Houd ook rekening met de soort sommen:" +
        "Bewerkingen" +
        " Getalbegrip\n" +
        " Kommagetallen\n" +
        " Breuken\n" +
        " Procenten\n" +
        " Verhoudingen\n" +
        " Meten\n" +
        " Tijd\n" +
        " Geld" +
        ", zorg dat de verhaaltjessom aansluit bij de soort som" +
        "Maak niet alleen sommen met getallen onder de 10, denk na over de groep, een groep 8 kan ook sommen boven de 10 of 100 hebben, maar een groep 6 misschien niet" +
        "Ik wil geen plaatjes" +
        "Maak ook sommen waar het antwoord een komme getal is" +
        "zet ALLEEN het verhaal neer, NIKS ANDERS ERBIJ" +
        "Als je een meet-som maakt, probeer dan ook te variëren in inhoudsmaten dus bijvoorbeeld centimeters en meters, of meters en liters" +
        "De sommen van groep 8 moeten ECHT HEEL moeilijk zijn"
        // `Om elk getal in het verhaal moet je <strong heen zetten, GEEN STERREN, dus bijvoorbeeld: Hannah heeft <strong 8 </strong bananend ... etc`
    let question = "dit is een voorbeeldvraag dus maak hem simpel"
    try {
        if (req.body) {

            let group = req.body.group
            let type = req.body.type
            const relevantDocs = await vectorStore.similaritySearch(`Als er voor deze groep: ${group}, verhaaltjesommen te vinden zijn in het document. Vat dan de moeilijkheidsgraad samen`,5);
            const context = relevantDocs.map(doc => doc.pageContent).join("\n\n");
            question = `Maak een vraag voor een kind uit groep ${group}. Maak er een verhaaltjessom van in de vorm van een ${type}-som. Zorg dat je aansluit op de belevingswereld van de kinderen. En op het niveau van de kinderen.Gebruik deze context: ${context} voor inspiratie van de soort sommen`

            const messages = [
                new SystemMessage(system),
            ]
            messages.push(new HumanMessage(question))

            const stream = await model.stream(messages);

            for await (const chunk of stream){
                await new Promise(resolve => setTimeout(resolve, 100))
                res.write(chunk.content)
            }
            res.end()
        } else {
            res.status(400).json({ error: 'Missing q in request body' });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Something went wrong' });
    }
});

app.post('/check', async (req, res) => {
    let systemQuestion = "haal de som uit het verhaal, bedenk goed wat voor soort som dit moet zijn, schrijf ALLEEN de som op, zonder antwoord, zet alle getallen en TEKENS achter elkaar zonder spaties, als het een keersom is doe dan * in plaats van x"
    let answer = "je hebt nog geen antwoord gegeven"
    let answerQuestion = ""
    try {
        if (req.body) {
            const laatsteVraag = req.body.question;
            let question = `Haal de som uit dit verhaal: ${laatsteVraag}`;
            const messages = [
                new SystemMessage(systemQuestion),
                new HumanMessage(question)
            ]

            let som = await model.invoke(messages);
            async function haalAntwoordOp(expr) {
                const encodedExpr = encodeURIComponent(expr);
                const url = `http://api.mathjs.org/v4/?expr=${encodedExpr}`;
                console.log(url);
                try {
                    const response = await fetch(url);
                    const antwoord = await response.text();
                    console.log('Antwoord van Math.js API:', antwoord);
                    return antwoord;
                } catch (error) {
                    console.error('Fout bij ophalen van antwoord:', error);
                    return null;
                }
            }
            answerQuestion = await haalAntwoordOp(som.content);
            // res.json(answerQuestion);

        } else {
            res.json({answer: "Je hebt nog geen som opgehaald"})
        }
        if (req.body.answer) {
            let system = "Controleer het antwoord op de vraag, doe dit als een basisschool leerkracht" +
                "Geef er ook een uitleg bij als het goed of fout is, hoe moet de som opgelost worden" +
                "Spreek de gebruiker aan als: je hebt het fout want ..." +
                "Spreek het aan als een kind, wees positief, wees een leerkracht, help ze om beter te worden" +
                "zeg dingen als: goed geprobeerd maar.... of iets in die richting, zeg niet je hebt het fout! gebruik positieve taal, maar wees wel realistisch, als ze het goed hebben mag je dat ook meteen zeggen" +
                "Het juiste antwoord wat je krijgt is altijd alleen een getal, kijk daarom goed naar de vormgeving van de vraag, wordt er gevraagd om eenheden en is het kommagetal misschien hetzelfde als een antwoord met rest" +
                "Als er een onzin antwoord wordt gegeven zoals hihihi of hoi of ALLEEN woorden, zeg dan dat ze het opnieuw moeten proberen, geef ze dan nog niet het antwoord" +
                "Er moet een antwoord worden gegeven, ja of nee is ook NIET goed, dan MOET de gebruiker het nog eens proberen en dan mag je het antwoord NIET geven"

            let answerUser = req.body.answer
            const laatsteVraag = [...req.body.history].reverse().find(item => item.ai)?.ai;
            let question = `Leg de som uit van dit verhaal: ${laatsteVraag}, het antwoord is ${answerQuestion}. Dit is het antwoord van de gebruiker: ${answerUser}, leg uit of ze het goed of fout hebben gedaan`;


            const messages = [
                new SystemMessage(system),
            ]
            if(req.body.history){
                let history = req.body.history
                console.log(history)
                for (const item of history) {
                    if (item.human) {
                        messages.push(new HumanMessage(item.human));
                    } else if (item.ai) {
                        messages.push(new AIMessage(item.ai));
                    }
                }

            }
            messages.push(new HumanMessage(question))
            const stream = await model.stream(messages);

            for await (const chunk of stream){
                await new Promise(resolve => setTimeout(resolve, 100))
                res.write(chunk.content)
            }
            res.end()

        } else {
            res.json("Je hebt nog geen antwoord gegeven")
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Something went wrong' });
    }
});
//
// app.post('/chat', async (req, res) => {
//     try {
//         if (req.body.q) {
//
//             let q = req.body.q;
//             const messages = [
//                 new SystemMessage("You are a friendly chatbot, that likes to answer questions"),
//             ]
//             if(req.body.history){
//                 let history = req.body.history
//                 console.log(history)
//                 for(const {human, ai} of history){
//                     messages.push(new HumanMessage(human))
//                     messages.push(new AIMessage(ai))
//                 }
//             }
//             messages.push(new HumanMessage(`${q}`))
//             console.log(messages)
//             const chat = await model.invoke(messages);
//             const stream = await model.stream(messages);
//
//             // console.log(chat.content);
//             // res.json({ chat: chat.content });
//
//             for await (const chunk of stream){
//                 await new Promise(resolve => setTimeout(resolve, 100))
//                 res.write(chunk.content)
//             }
//             res.end()
//         } else {
//             res.status(400).json({ error: 'Missing q in request body' });
//         }
//     } catch (error) {
//         console.error(error);
//         res.status(500).json({ error: 'Something went wrong' });
//     }
// });
//


app.get('/', (req, res) => {
    res.send('Hello world!');
});

app.listen(port, () => {
    console.log(`Server draait op http://localhost:${port}`);
});

app.use(cors())


