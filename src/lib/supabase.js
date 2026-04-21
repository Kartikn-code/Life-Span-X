import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://wlagxpldewuyhnnlcxxa.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndsYWd4cGxkZXd1eWhubmxjeHhhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3OTM0ODgsImV4cCI6MjA5MjM2OTQ4OH0.FACg6YVZvnu6pA8p15jOkZwgFCwtC6gNtdLNd4LZFOY';

export const supabase = createClient(supabaseUrl, supabaseKey);
