import time
from amico_stt import stt
from amico_listen import record_audio

#import warnings
#warnings.filterwarnings("ignore", message=r".*list_audio_backends.*deprecated.*", category=UserWarning)
#warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.custom_fwd.*deprecated.*", category=FutureWarning)

#from audio_input import record_audio
#from transcriber import transcribe_and_detect_language
#from voice_auth import extract_voiceprint, check_voice_match, insert_new_user_with_voiceprint,store_voiceprint_if_needed,confirm_user_by_voice,get_user_name_by_id
#from voice_utils import speak_text
#import subprocess
#from gpt_utils import extract_name_from_text,translate_text,ask_gpt_with_web
#from user_handling import handle_known_user, handle_unknown_user, handle_possible_user


def main():
    print("🤖 AMICO is running. Press Ctrl+C to stop.")

    try:
        while True:
            print("🎙️ AMICO v0.2")               
            audio_path = record_audio()
            input("Press Enter to continue...") 
            print("📝 Transcribing...")
            text,language = stt(audio_path)
            print(f"🗣️ You said: {text}")
            print(f"🌐 Detected language: {language}")
            input("Press Enter to continue...") 
            print("🗣️ Temp store of "You said" for storage purpose")
            user_said = text
            input("Press Enter to continue...")
            print("📝 Extracting Voiceprint...")
            #voiceprint = extract_voiceprint(audio_path)
            #print("🔊 Voiceprint vector:")
            #print(f"🌐 {voiceprint})
            #print("🧮 Shape:", voiceprint.shape)
            #Mostramos si la voiceprint ha sido extraída o no
            if voiceprint is not None and voiceprint.nelement() > 0:
                print("✅ Voiceprint extracted")
            #else:
                print("⚠️ No voiceprint extracted")
            input("Press Enter to continue...") 
            quit()
            #print("📝 Checking user...")
            #user_id,similarity = check_voice_match(voiceprint)            
            #print(f"📝 : {similarity}")
            #input("Press Enter to continue...") 
            #if similarity < 0.99: # it should be 0.45
                #print("🆕 New user")
                #input("Press Enter to continue...") 
                #handle_unknown_user(voiceprint)
                
                
                                
            #elif 0.1 <= similarity < 0.2:    #must be 0.45 to 0.75
                #print("🤔 Possible known user")
                #input("Press Enter to continue...") 
                #suspected_name = get_user_name_by_id(user_id) 
                #confirmed = confirm_user_by_voice(suspected_name, language) 
                #if confirmed:
                        #user_id = user_id  # reuse the same ID
                        #final_reply = chat_with_gpt(messages, language=language)
                #else:
                        # re-run matching logic using the new input
                        #user_id, similarity = check_voice_match(voiceprint)
                        #quit()
            #else:
                #print(f"✅ Known user: {user_id}")
                #input("Press Enter to continue...")
                #handle_known_user(user_id,voiceprint)
                #print("💬 Responding to the original input...")
                #prompt = [{"role": "user", "content": user_input_text}]
                #final_reply = ask_gpt_with_web(prompt, language=language)
                #print(f"🤖 GPT reply: {final_reply}")
                #speak_text(final_reply)
                
                
                









#except Exception as e:
 #   print(f"❌ Error extracting voiceprint: {e}")



            #print("🔐 Authenticating user...")
            #user_id, user_name = authenticate_user(audio_path, language)

            #print(f"👤 User: {user_name} (ID: {user_id})")

            #print("💬 Generating response...")
            #response = chat_with_gpt(text, user_id, user_name)

            #print(f"🤖 AMICO: {response}")
            #speak_text(response)

            #log_user_input(user_id, text, response)

            #print("⏱️ Waiting for next input...\n")
            #time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Shutting down AMICO.")

if __name__ == "__main__":
    main()
