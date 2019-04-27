const fetch = require('node-fetch')
const fs = require('fs')
const path = require('path');
const os = require('os');

   // output file in the same folder
const filename = path.join(__dirname, ' deeds_sonny_user.csv');
const output = []; // holds all rows of data

let cursor = -1
obj = []

const delay = (interval) => {
    return new Promise((resolve) => {
        setTimeout(resolve, interval);
    });
};

async function callAPI(){
//count = 0
while(cursor != 0 ){
 console.log(`The cursor number is ${cursor}`)
 var url = `https://api.twitter.com/1.1/followers/list.json?&skip_status=true&include_user_entities=false&user_id=980177523001700352&count=200&cursor=${cursor}`;
 //count++
 //console.log(count)
 await fetch(url, {
 method: 'GET',
 headers:{
   'Content-Type': 'application/json',
   'authorization' : 'authorization code'
 }
 }).then(res => res.json())
   .then(response => {
     //console.log(response)
     obj = [...obj, ...response.users]
     cursor = response.next_cursor
   })
 .catch(error => console.error('Error:', error));
  await delay(60000);
  }
  obj.forEach((d) => {
        const row = []; // a new array for each row of data
        row.push(d.id)
        //console.log("user name is " + d.name[0])
        row.push(d.screen_name);
        row.push((d.url)?1:0);
        row.push((d.profile_banner_url)?1:0);
        row.push((d.location)?1:0);
        row.push((d.has_extended_profile)?1:0);
        row.push(d.followers_count);
        row.push(d.friends_count);

        output.push(row.join()); // by default, join() uses a ','
      });
  fs.writeFileSync(filename, output.join(os.EOL));
 }

callAPI()
