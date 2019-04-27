const fetch = require('node-fetch')
const csv = require('csv-parser')
const fs = require('fs')
const path = require('path');
const os = require('os');
const results = [];
const filename = path.join(__dirname, 'joma_tweets.csv');
const output = []

array = [2996506014, 36022454, 40394313]

function calTweetsNumber(response){
  if(!response[0]){
    return -1
  }
  return response.length
}

function totalRetweetCount(response){
  if(!response[0]){
    return -1
  }
  let total = 0
  for(j = 0 ; j < response.length; j++){
    total += response[j].retweet_count
  }
  return total
}

function numberOfDuplicate(response){
  let count = 0;
  if(!response[0]){
    return -1
  }
  for(let i = 0 ; i < response.length; i++){
    for(let j = i+1; j < response.length; j++){
      if(i != j && response[i].text == response[j].text){
        count++;
      }
    }
  }

  return count;
}

function tweestOnlyHasUrl(response){
  let count = 0;
  if(!response[0]){
    return -1
  }
  for(let c = 0 ; c < response.length; c++){
    if(response[c].text.length == 0 && response[c].entities.urls.length >=1 ){
      count++
    }
  }
  return count
}

function tweetContainsHashTag(response) {
  let count = 0
  if(!response[0]){
    return -1
  }
  for (const item of response) {
    //console.log(item)
    let hashtags = item.entities.hashtags
    if(hashtags.length >=1 ){
      count++
    }
  }
  return count
}



//The thing we need to do:
//1. Find whether the tweets has a duplicate
//2. Active time(The Attribute "Created At")
//3. Contains hashtag("hashtag attribute")
//4. Is retweeted
//5. publish with only urls
//6. Sign in device('Log in with Twitter') => API reference
//i = 0
async function call(results){
  //console.log("result is " + results)
  let i = 0
  while(i < results.length){
    console.log("i = " + i)
    console.log(results[i])
    var url = `https://api.twitter.com/1.1/statuses/user_timeline.json?user_id=${results[i]}&count=200`;
    await fetch(url, {
    method: 'GET',
    headers:{
      'Content-Type': 'application/json',
      'authorization' : 'authorization code'
    }
    }).then(res => res.json())
      .then(response => {
        console.log("Response length = " + response.length)
         const row = []
         //console.log(totalRetweetCount(response))
         row.push(results[i])
         row.push(calTweetsNumber(response))
         row.push(totalRetweetCount(response))
         row.push(numberOfDuplicate(response))
         row.push(tweestOnlyHasUrl(response))
         row.push(tweetContainsHashTag(response))
         output.push(row.join())
         i++
      })
    .catch(error => console.error('Error:', error));
  }
  fs.writeFileSync(filename, output.join(os.EOL))
}

fs.createReadStream('joma_user.csv')
  .pipe(csv())
  .on('data', (data) => results.push(data.user_id))
  .on('end', () => {
    //console.log("results length = " + results.length)
    call(results)
  });
