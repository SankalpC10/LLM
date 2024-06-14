//Could not find a declaration file for module 'dotenv' -> 
//https://stackoverflow.com/questions/58211880/uncaught-syntaxerror-cannot-use-import-statement-outside-a-module-when-import


//# Import required dependencies from npm and load the API key
// @ts-ignore
import dotenv from 'dotenv';

dotenv.config({ path: '.env' });

const keys = process.env; // read API keys from .env

import { 
    Document, 
    VectorStoreIndex, 
    SimpleDirectoryReader 
} from "npm:llamaindex@0.1.8"

//# Load our data from a local directory
const documents = await new SimpleDirectoryReader()
    .loadData({directoryPath: "./data"})
